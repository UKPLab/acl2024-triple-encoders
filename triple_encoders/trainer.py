import csv
import gzip
import logging
from os.path import exists
import random
from typing import List, Callable, Dict, Type

import transformers
from sentence_transformers import InputExample, models, SentenceTransformer, losses
from triple_encoders.evaluator import EmbeddingsTripleSimilarityEvaluator
from triple_encoders.losses import CosineSimilarityTripleEncoderLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

logger = logging.getLogger(__name__)


class TripleEncoderTrainer:
    """
    Trainer for  Triple-Encoder models
    """

    def __init__(self,
                 base_model_name_or_path: str = None,
                 batch_size: int = 16,
                 observation_window=5,
                 speaker_token: bool = True,
                 num_epochs: int = 1,
                 warmup_steps: int = 10000,
                 model_args: Dict = {},
                 ):

        self.base_model_name_or_path = base_model_name_or_path
        self.model_save_path = None
        self.batch_size = batch_size
        self.observation_window = observation_window
        self.num_epochs = num_epochs
        self.speaker_token = speaker_token
        self.warmup_steps = warmup_steps

        tokens = ["[BEFORE]", "[AFTER]", "[B1]", "[B2]"]
        if self.speaker_token:
            tokens += ["[O]", "[E]"]

        base_model = models.Transformer(base_model_name_or_path, model_args=model_args)

        base_model.tokenizer.add_tokens(tokens, special_tokens=True)
        base_model.auto_model.resize_token_embeddings(len(base_model.tokenizer))

        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(base_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)

        self.model = SentenceTransformer(modules=[base_model, pooling_model])

        logger.info(f"Model {base_model_name_or_path} loaded")

    def generate_datasets(self, train_dataset, validation_dataset, evaluation_dataset, mixed_with_nli=True):
        """
        Generate datasets for Contextual Curved Contrastive Learning (C3L) objective and NLI objective
        :param train_dataset: dialog dataset for training
        :param validation_dataset: dialog dataset for validation
        :param evaluation_dataset: dialog dataset for evaluator
        :param mixed_with_nli: if True, mix the Curved Learning objective with NLI objective
        :return:
        """

        # get random utterances for negative examples
        logger.info("Generating datasets for Curved Learning objective")

        self.random_utterances = [utterance for dialog in train_dataset for utterance in dialog]
        logger.info("Generated {} random utterances".format(len(self.random_utterances)))

        self.dialog_train_dataset = self._generate_C3L_data(train_dataset)
        logger.info("Generated {} training Inputs".format(len(self.dialog_train_dataset)))

        self.dialog_validation_dataset = self._generate_C3L_data(validation_dataset)
        logger.info("Generated {} validation Inputs".format(len(self.dialog_validation_dataset)))

        self.dialog_evaluation_dataset = self._generate_C3L_data(evaluation_dataset)
        logger.info("Generated {} evaluator Inputs".format(len(self.dialog_evaluation_dataset)))

        logger.info("Completed generating datasets for Contextualized Curved Learning objective")

        logger.info("Generating datasets NLI dataset")

        self.mixed_with_nli = mixed_with_nli

        if mixed_with_nli:
            # source from Sentence Transformers
            # https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/other/training_multi-task.py

            # Check if dataset exsist. If not, download and extract  it
            nli_dataset_path = 'datasets/AllNLI.tsv.gz'

            if not exists(nli_dataset_path):
                raise Exception(
                    f"Couldn't find nli dataset file at path: {nli_dataset_path}"
                    "\nPlease run the following commands to download and save the required file."
                    "\n$ wget https://sbert.net/datasets/AllNLI.tsv.gz"
                    "\n$ mv AllNLI.tsv.gz ./datasets/AllNLI.tsv.gz"
                )

            logging.info("Read AllNLI train dataset")
            label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
            train_nli_samples = []
            with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
                reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
                for row in reader:
                    if row['split'] == 'train':
                        label_id = label2int[row['label']]
                        train_nli_samples.append(
                            InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))

            self.nli_dataset = train_nli_samples

        logger.info("Generated all datasets, ready to train")

    def _generate_C3L_data(self, text_set):
        """
        creates the C3L triplets for training triple encoders
        :param text_set: List of List of utterances (strings) i.e. provide (List of dialogues)
        :return:
        """
        # create CCL bi-encoder dict
        bi_dict = {}
        reverse_list = list(range(self.observation_window - 1))[::-1]
        for i in range(1, self.observation_window):
            bi_dict[reverse_list[i - 1]] = (self.observation_window - i) / self.observation_window

        # create triple dict from CCL bi-encoders scores
        triplets_dict = {}
        for key in bi_dict:
            for key2 in bi_dict:
                if key < key2:
                    triplets_dict[(key, key2)] = bi_dict[key] + bi_dict[key2]

        # normalizing the triplets
        max_triplet = max(triplets_dict.values())
        min_triplet = min(triplets_dict.values())
        min_value = 0.2
        max_value = 1.0
        for key in triplets_dict:
            triplets_dict[key] = (triplets_dict[key] - min_triplet) / (max_triplet - min_triplet) * (
                        max_value - min_value) + min_value
        dialog_dataset = []

        # add speaker tokens for conversational sequence modeling
        if self.speaker_token:
            player_even = "[E] "
            player_odd = "[O] "
        else:
            player_even = ""
            player_odd = ""
        for dialog_i in range(len(text_set)):
            # slice through the dialog with windows of size self.observation_window
            for i in range(0, len(text_set[dialog_i]) - self.observation_window):
                dialog_window = text_set[dialog_i][i:i + self.observation_window]

                # Determine the player token of the last utterance
                last_utterance_token = player_even if len(dialog_window) % 2 == 0 else player_odd

                for (context1_idx, context2_idx), target_cosine in triplets_dict.items():
                    # Determine player tokens based on the last utterance
                    if last_utterance_token == player_even:
                        player1_token = player_odd if context1_idx % 2 == 0 else player_even
                        player2_token = player_odd if context2_idx % 2 == 0 else player_even
                    else:
                        player1_token = player_even if context1_idx % 2 == 0 else player_odd
                        player2_token = player_even if context2_idx % 2 == 0 else player_odd

                    inp_example = InputExample(
                        texts=["[B1] " + player1_token + "[BEFORE] " + dialog_window[context1_idx],
                               "[B2] " + player2_token + "[BEFORE] " + dialog_window[context2_idx],
                               "[AFTER] " + dialog_window[-1]],
                        label=target_cosine)

                    dialog_dataset.append(inp_example)

                # Adding hard negatives with random utterances
                for context_idx in range(4):  # u_1 to u_4
                    # Determine player token based on the last utterance
                    if last_utterance_token == player_even:
                        player_token = player_odd if context_idx % 2 == 0 else player_even
                    else:
                        player_token = player_even if context_idx % 2 == 0 else player_odd

                    player_token_random = random.choice([player_even, player_odd])

                    random_utterance1 = random.choice(self.random_utterances)
                    random_utterance2 = random.choice(self.random_utterances)

                    inp_example = InputExample(
                        texts=["[B1] " + player_token + "[BEFORE] " + dialog_window[context_idx],
                               "[B2] " + player_token_random + "[BEFORE] " + random_utterance1,
                               "[AFTER] " + dialog_window[-1]],
                        label=0.0)  # Hard negatives have 0 cosine similarity

                    dialog_dataset.append(inp_example)

                    inp_example = InputExample(
                        texts=["[B1] " + player_token_random + "[BEFORE] " + random_utterance2,
                               "[B2] " + player_token + "[BEFORE] " + dialog_window[context_idx],
                               "[AFTER] " + dialog_window[-1]],
                        label=0.0)  # Hard negatives have 0 cosine similarity

                    dialog_dataset.append(inp_example)

                    # Determine player token based on the last utterance
                    player_token_random1 = random.choice([player_even, player_odd])
                    player_token_random2 = random.choice([player_even, player_odd])

                    random_utterance1 = random.choice(self.random_utterances)
                    random_utterance2 = random.choice(self.random_utterances)

                    inp_example = InputExample(
                        texts=["[B1] " + player_token_random1 + "[BEFORE] " + random_utterance2,
                               "[B2] " + player_token_random2 + "[BEFORE] " + random_utterance1,
                               "[AFTER] " + dialog_window[-1]],
                        label=0.0)  # Hard negatives have 0 cosine similarity

                    dialog_dataset.append(inp_example)

            return dialog_dataset

    def train(self,
              model_save_path: str,
              batch_size: int = 16,
              epochs: int = 1,
              steps_per_epoch=None,
              scheduler: str = 'WarmupLinear',
              warmup_steps: int = 10000,
              optimizer_class: Type[Optimizer] = transformers.AdamW,
              optimizer_params: Dict[str, object] = {'lr': 2e-5},
              weight_decay: float = 0.01,
              evaluation_steps: int = 0,
              save_best_model: bool = True,
              max_grad_norm: float = 1,
              use_amp: bool = False,
              callback: Callable[[float, int, int], None] = None,
              show_progress_bar: bool = True,
              checkpoint_path: str = None,
              checkpoint_save_steps: int = 500,
              checkpoint_save_total_limit: int = 0
              ):
        """
        # adjusted from the Sentence Transformer package
        source: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py


        :param model_save_path: Output path for the model
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluator files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluator.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        :param checkpoint_path: Folder to save checkpoints during training
        :param checkpoint_save_steps: Will save a checkpoint after so many steps
        :param checkpoint_save_total_limit: Total number of checkpoints to store
        :return: 
        """

        if not model_save_path:
            raise ValueError("model_save_path must be specified")

        if not getattr(self, "dialog_train_dataset"):
            raise ValueError("You need to first call generate_datasets() before calling train()")

        train_dataloader_dialogue = DataLoader(self.dialog_train_dataset, shuffle=True, batch_size=batch_size,
                                               collate_fn=lambda x: tuple(
                                                   x_.to('cuda') for x_ in default_collate(x)))

        train_loss_dialogue = CosineSimilarityTripleEncoderLoss(model=self.model)

        train_objectives = [(train_dataloader_dialogue, train_loss_dialogue)]

        if self.mixed_with_nli:
            train_loss_nli = losses.ContrastiveLoss(model=self.model)
            train_dataloader_nli = DataLoader(self.nli_dataset, shuffle=True, batch_size=batch_size,
                                               collate_fn=lambda x: tuple(
                                                   x_.to('cuda') for x_ in default_collate(x)))

            train_objectives.append((train_dataloader_nli, train_loss_nli))

        evaluator = EmbeddingsTripleSimilarityEvaluator.from_input_examples(self.dialog_validation_dataset, name='test')

        _ = self.model.to('cuda')

        # Train the model
        self.model.fit(train_objectives=train_objectives,
                        evaluator=evaluator,
                        epochs=epochs,
                        evaluation_steps=evaluation_steps,
                        warmup_steps=warmup_steps,
                        output_path=model_save_path,
                        optimizer_class=optimizer_class,
                        optimizer_params=optimizer_params,
                        weight_decay=weight_decay,
                        scheduler=scheduler,
                        show_progress_bar=show_progress_bar,
                        use_amp=use_amp,
                        callback=callback,
                        checkpoint_path=checkpoint_path,
                        checkpoint_save_steps=checkpoint_save_steps,
                        checkpoint_save_total_limit=checkpoint_save_total_limit,
                        max_grad_norm=max_grad_norm,
                        steps_per_epoch=steps_per_epoch,
                        save_best_model=save_best_model)

        logger.info(f"Training finished. You can now load the model from {model_save_path}")




import logging
import pickle
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from triple_encoders.TripleEncodersForSequenceModeling import TripleEncodersForSequenceModeling, Sequence

import logging

logger = logging.getLogger(__name__)

class TripleEncodersForConversationalSequenceModeling(TripleEncodersForSequenceModeling):
    """
    triple_encoders for Sequence Modeling
    """
    def __init__(self,
                 model_name_or_path: Optional[str] = None
                 ):
        """
        :param model_name_or_path: Path to the model or the model name from huggingface.co/models
        """
        if not model_name_or_path:
            raise ValueError("Please provide a model name or path")

        super().__init__(model_name_or_path)

        self.candidates = None
        self.candidates_embeddings = None

    def encode_context_with_speaker_token(self, context: List[str], dialog_partner: bool):
        """
        :param context: List of strings
        :param dialog_partner: If True, the last utterance in the context will be considered as odd
        :return: returns two tensors (b1 and b2), each of shape (len(context), embedding_dim)
        """
        combined_context = []

        context = context[::-1]  # Reverse the context to encode the most recent utterance first

        # Collect utterances with their respective speaker tokens
        for i, utterance in enumerate(context):
            speaker_token = "[E]" if (i % 2 == 0) != dialog_partner else "[O]"
            combined_context.append(f"[B1] {speaker_token} [BEFORE] {utterance}")
            combined_context.append(f"[B2] {speaker_token} [BEFORE] {utterance}")

        # Encode all utterances in a single batch operation
        encoded_context = self.model.encode(combined_context,
                                            show_progress_bar=False,
                                            convert_to_tensor=True)

        # Split the batch encoded embeddings back into B1 and B2 spaces
        encoded_context_b1 = encoded_context[::2]  # Even indices for B1
        encoded_context_b2 = encoded_context[1::2]  # Odd indices for B2

        # reverse the order of the encoded context
        encoded_context_b2 = torch.flip(encoded_context_b2, (0,))
        encoded_context_b1 = torch.flip(encoded_context_b1, (0,))

        return encoded_context_b1, encoded_context_b2

    def contextualize_sequence(self,
                               context: List[str],
                               last_dialog_partner : Optional[bool] = False,
                               k_last_rows: Optional[int] = None,
                               ):
        """
        :param context: List of strings that will be used as context
        :param last_dialog_partner: If set to true, the last utterance in the context will be considered as odd
        :param k_last_rows: if set, only the last k rows of the context (emerging triangle of states in triple encoders) will be used
        e.g. k_last_rows takes the last k utterances in the [B2] space and contextualize them with the entire [B1] space
        :return: SequenceObject with the context embeddings
        """

        encoded_context_b1, encoded_context_b2 = self.encode_context_with_speaker_token(context, last_dialog_partner)

        contextualized_embeddings, indices = self._contextualize_sequence(encoded_context_b1,
                                                                encoded_context_b2,
                                                                k_last_rows)

        sequence_object = Sequence(context,
                                   contextualized_embeddings,
                                   encoded_context_b1,
                                   encoded_context_b2,
                                   k_last_rows=k_last_rows,
                                   contextualized_embeddings_indices=indices)

        return sequence_object




    def contextualize_utterance(self, utterance: str,
                                sequence_object: Optional[Sequence] = None,
                                dialog_partner: Optional[bool] = False
                                ):
        """
        This method adds an utterance to the context and updates the context embeddings
        if precompute_top_p is set, the top p *100% candidates will be precomputed

        :param utterance: new utterance to add to the context
        :param sequence_object: SequenceObject with the context embeddings
        :param dialog_partner: If True, the last utterance in the context will be considered as odd
        """

        if sequence_object is None:
            sequence_object = Sequence()

        if dialog_partner:
            speaker_token = "[O]"
        else:
            speaker_token = "[E]"

        utterance_b1 = [f"[B1] {speaker_token} [BEFORE]  " + utterance]
        utterance_b2 = [f"[B2] {speaker_token} [BEFORE]  " + utterance]

        utterance_embeddings_combined = self.model.encode(
            utterance_b1 + utterance_b2,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True).to(self.device)

        utterance_b1_encoded = utterance_embeddings_combined[0]
        utterance_b2_encoded = utterance_embeddings_combined[1]


        def update_context_embedding(context_embeddings, utterance_encoded):
            """
            if no context_embeddings are present, create a new context_embeddings
            else, update the context_embeddings with the new utterance embeddings
            """
            if context_embeddings is None:
                return torch.stack([utterance_encoded])
            else:
                return torch.cat([context_embeddings, torch.stack([utterance_encoded])])


        # Update context embeddings with the new utterance embeddings
        context_embeddings_b1 = update_context_embedding(sequence_object.context_embeddings_b1, utterance_b1_encoded)
        context_embeddings_b2 = update_context_embedding(sequence_object.context_embeddings_b2, utterance_b2_encoded)

        sequence_object.context_embeddings_b1 = context_embeddings_b1
        sequence_object.context_embeddings_b2 = context_embeddings_b2

        sequence_object.contextualized_embeddings, sequence_object.contextualized_embeddings_indices \
            = self._contextualize_utterance(context_embeddings_b1, utterance_b2_encoded)

        return sequence_object

    def evaluate_seq_dataset(self,
                             dialogues: List[List[str]],
                             history_lengths: List[int] =  [2,3,4,5,6,7,8,9,10],
                             last_k_rows: Optional[int] = None,
                             batch_size: int = 32):
        """
        This method evaluates the sequence model on a dataset of dialogues.
        :param dialogues: List of dialogues. Each dialogue is a list of utterances.
        :param history_lengths: List of history lengths to evaluate.
        :param last_k_rows: if set, only the last k rows of the context (emerging triangle of states in triple encoders) will be used
        e.g. k_last_rows takes the last k utterances in the [B2] space and contextualize them with the entire [B1] space
        :param batch_size: Batch size for bmm across number of dialogues in parallel in the evaluation.
        """
        batch_size = min(batch_size, len(dialogues))

        report_dict = {}

        # flatten dialog text and create two arrays for dialog_id and utterance_id in dialog dialog_test
        flattened_dialog_test = [item for sublist in dialogues for item in sublist]

        # add after token to flatten dialog
        flattened_dialog_test_a = ["[AFTER]  " + utterance for utterance in flattened_dialog_test]

        flattened_dialog_test_b_even_b1 = ["[B1] [E] [BEFORE] " + utterance for utterance in flattened_dialog_test]
        flattened_dialog_test_b_odd_b1 = ["[B1] [O] [BEFORE] " + utterance for utterance in flattened_dialog_test]

        flattened_dialog_test_b_even_b2 = ["[B2] [E] [BEFORE] " + utterance for utterance in flattened_dialog_test]
        flattened_dialog_test_b_odd_b2 = ["[B2] [O] [BEFORE] " + utterance for utterance in flattened_dialog_test]

        flattened_dialog_test_encoded_a = self.model.encode(flattened_dialog_test_a, show_progress_bar=False,
                                                       normalize_embeddings=True)

        flattened_history_encoded_b_even_b1 = self.model.encode(flattened_dialog_test_b_even_b1,
                                                                show_progress_bar=False)
        flattened_history_encoded_b_odd_b1 = self.model.encode(flattened_dialog_test_b_odd_b1,
                                                               show_progress_bar=False)

        flattened_history_encoded_b_even_b2 = self.model.encode(flattened_dialog_test_b_even_b2,
                                                                show_progress_bar=False)
        flattened_history_encoded_b_odd_b2 = self.model.encode(flattened_dialog_test_b_odd_b2,
                                                               show_progress_bar=False)

        dialog_id = [i for i, sublist in enumerate(dialogues) for item in sublist]
        utterance_id = [j for sublist in dialogues for j, item in enumerate(sublist)]

        # reshape to dialog test based on dialog_id and utterance_id because not all dialogs have the same length
        flattened_dialog_test_encoded_reshaped = []
        flattened_dialog_test_encoded_reshaped_history_even_b1 = []
        flattened_dialog_test_encoded_reshaped_history_odd_b1 = []
        flattened_dialog_test_encoded_reshaped_history_even_b2 = []
        flattened_dialog_test_encoded_reshaped_history_odd_b2 = []
        for i, dialog in enumerate(dialogues):
            flattened_dialog_test_encoded_reshaped.append(
                flattened_dialog_test_encoded_a[dialog_id.index(i):dialog_id.index(i) + len(dialog)])
            flattened_dialog_test_encoded_reshaped_history_even_b1.append(
                flattened_history_encoded_b_even_b1[dialog_id.index(i):dialog_id.index(i) + len(dialog)])
            flattened_dialog_test_encoded_reshaped_history_odd_b1.append(
                flattened_history_encoded_b_odd_b1[dialog_id.index(i):dialog_id.index(i) + len(dialog)])
            flattened_dialog_test_encoded_reshaped_history_even_b2.append(
                flattened_history_encoded_b_even_b2[dialog_id.index(i):dialog_id.index(i) + len(dialog)])
            flattened_dialog_test_encoded_reshaped_history_odd_b2.append(
                flattened_history_encoded_b_odd_b2[dialog_id.index(i):dialog_id.index(i) + len(dialog)])

        for history_length in history_lengths:
            report_dict[f"H{history_length}_mean_rank"] = None

        for history_length in history_lengths:
            print(f'history length: {history_length}')
            # create history
            history = []
            next_utterances_id_same_length = []
            flattened_next_utterances_same_length = []
            iteration = 0
            for i, dialog in enumerate(dialogues):
                if len(dialog) > history_length:
                    current_history_even_b1 = flattened_dialog_test_encoded_reshaped_history_even_b1[i][
                                              :history_length][::-1]
                    current_history_odd_b1 = flattened_dialog_test_encoded_reshaped_history_odd_b1[i][:history_length][
                                             ::-1]
                    current_history_even_b2 = flattened_dialog_test_encoded_reshaped_history_even_b2[i][
                                              :history_length][::-1]
                    current_history_odd_b2 = flattened_dialog_test_encoded_reshaped_history_odd_b2[i][:history_length][
                                             ::-1]

                    current_history_b1 = []
                    current_history_b2 = []
                    for dist in range(len(current_history_even_b1)):
                        if dist % 2 == 0:
                            current_history_b1.append(current_history_odd_b1[dist])
                            current_history_b2.append(current_history_odd_b2[dist])
                        else:
                            current_history_b1.append(current_history_even_b1[dist])
                            current_history_b2.append(current_history_even_b2[dist])
                    current_history_b1 = current_history_b1[::-1]
                    current_history_b2 = current_history_b2[::-1]

                    current_history_b1 = torch.tensor(current_history_b1)
                    current_history_b2 = torch.tensor(current_history_b2)

                    contextualized_embeddings, _ = self._contextualize_sequence(current_history_b1, current_history_b2,
                                                                               k_last_rows=last_k_rows)

                    history.append(contextualized_embeddings)

                    flattened_next_utterances_same_length.append(flattened_dialog_test_encoded_reshaped[i]
                                                                 [history_length])
                    next_utterances_id_same_length.append(iteration)
                    iteration += 1

                context_embeddings_batch = torch.nn.functional.normalize(torch.stack(history),
                                                                         p=2, dim=2).to(self.device)
                self.candidates_embeddings = torch.from_numpy(np.array(flattened_next_utterances_same_length)).to(
                    self.device)

                ranks = self._sequence_model_batch(context_embeddings_batch, next_utterances_id_same_length,
                                                   batch_size=batch_size)

                report_dict[f"H{history_length}_mean_rank"] = np.mean(ranks)

        df = pd.DataFrame(report_dict, index=[0])
        return df








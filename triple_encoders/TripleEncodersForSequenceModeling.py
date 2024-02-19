import logging
import pickle
from typing import Optional, List, Union

import numpy as np
import pandas as pd
import torch
from triple_encoders.TripleEncoders import TripleEncoders

import logging

logger = logging.getLogger(__name__)

class Sequence:
    """
    Sequence object to store the context and the computed scores for the sequence modeling of candidates
    """
    def __init__(self,
                 context_utterances: Optional[List[str]] = None,
                 contextualized_embeddings: Optional[torch.Tensor] = None,
                 context_embeddings_b1: Optional[torch.Tensor] = None,
                 context_embeddings_b2: Optional[torch.Tensor] = None,
                 computed_scores: Optional[torch.Tensor] = None,
                 computed_scores_indices: Optional[torch.Tensor] = None,
                 sum_computed_scores: Optional[torch.Tensor] = None,
                 precompute: Optional[bool] = False,
                 k_last_rows: Optional[int] = None,
                 contextualized_embeddings_indices: Optional[List[int]] = None,
                 ):
        """

        :param contextualized_embeddings: contextualized representations for next sequence modeling step
        :param context_embeddings_b1:  embeddings in B1 space
        :param context_embeddings_b1: embeddings in B1 space
        :param computed_scores: computed scores from the previous sequence modeling step (one score for each candidate)
        :param computed_scores_indices: embedding indices of the previously computed scores (important for k_last_rows)
        :param sum_computed_scores: sum of the computed scores (aggregated over all query vectors)
        :param precompute: logic to check if scores are already computed
        :param k_last_rows: if set, only the last k rows of the context
        (emerging triangle of states in triple encoders: [B1]: columns, [B2] rows) will be used
        e.g. k_last_rows takes the last k utterances in the [B2] space and contextualize them with the entire [B1] space
        :param contextualized_embeddings_indices: indices of the contextualized embeddings that will be computed (important for k_last_rows)
        """
        self.context_utterances = context_utterances
        self.contextualized_embeddings = contextualized_embeddings
        self.context_embeddings_b1 = context_embeddings_b1
        self.context_embeddings_b2 = context_embeddings_b2
        self.computed_scores = computed_scores
        self.sum_computed_scores = sum_computed_scores
        self.contextualized_embeddings_indices = contextualized_embeddings_indices
        self.computed_scores_indices = computed_scores_indices
        self.precompute = precompute
        self.k_last_rows = k_last_rows


class TripleEncodersForSequenceModeling(TripleEncoders):
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

    def _contextualize_sequence(self, context_embeddings_b1, context_embeddings_b2,
                               k_last_rows: Optional[int] = None):
        """
        This function takes the context embeddings and creates the sequence with speaker tokens
        :param context_embeddings_b1: context embeddings in B1 space
        :param context_embeddings_b2: context embeddings in B2 space
        :param k_last_rows: if set, only the last k rows of the context
        (emerging triangle of states in triple encoders: [B1]: columns, [B2] rows) will be used
        e.g. k_last_rows takes the last k utterances in the [B2] space and contextualize them with the entire [B1] space
        :return: context_embeddings: contextualized representations, indices: indices of utterance pairs
        """
        sequence_length = len(context_embeddings_b1)
        if k_last_rows:
            k_last_rows_indices = list(range(sequence_length - k_last_rows, sequence_length))
        else:
            k_last_rows_indices = list(range(sequence_length))

        contextualized_embeddings = []
        indices = []
        for k in range(sequence_length):
            for l in range(sequence_length):
                if k < l and l in k_last_rows_indices:
                    stacked_tensors = torch.stack([context_embeddings_b1[k], context_embeddings_b2[l]], dim=0)
                    contextualized_embeddings.append(torch.mean(stacked_tensors, dim=0))
                    indices.append((k, l))

        context_embeddings = torch.stack(contextualized_embeddings, dim=0)

        # normalize embeddings
        context_embeddings = torch.nn.functional.normalize(context_embeddings, p=2, dim=1)

        return context_embeddings, indices

    def _contextualize_utterance(self, context_embeddings_b1, new_context_embedding_b2):
        """
        This function takes the context embeddings and creates the sequence with speaker tokens
        :param context_embeddings_b1: context embeddings in B1 space
        :param new_context_embedding_b2: new context embeddings in B2 space
        :return:
        context_embeddings: contextualized representations
        indices: indices of utterance pairs
        """
        sequence_length = len(context_embeddings_b1)
        if sequence_length == 0:
            raise ValueError("Create a context first, triple encoders requires minimum 2 utterances in the context")
        contextualized_embeddings = []
        indices = []
        new_index_b2 = sequence_length
        for k in range(sequence_length):
            stacked_tensors = torch.stack([context_embeddings_b1[k], new_context_embedding_b2], dim=0)
            contextualized_embeddings.append(torch.mean(stacked_tensors, dim=0))
            indices.append((k, new_index_b2))

        context_embeddings = torch.stack(contextualized_embeddings, dim=0)

        # normalize embeddings
        context_embeddings = torch.nn.functional.normalize(context_embeddings, p=2, dim=1)

        return context_embeddings, indices




    def load_candidates_from_strings(self, candidates: List[str],
                                     output_directory_candidates_dump: str = None):
        """
        :param candidates: List of candidate strings without any special tokens (will be added automatically)
        :param output_directory_candidates_dump: If set, the candidates will be stored as numpy array and as pickle list
        :param special_tokens: List of tokens that will be added to the candidates (same length as candidates)
                If None: [AFTER] will be added to the candidates (as in the original paper)
        :return: None
        """

        self.candidates = candidates

        candidates = ["[AFTER]  " + candidate for candidate in candidates]

        candidates_embeddings = self.model.encode(candidates,
                                                  show_progress_bar=True,
                                                  convert_to_tensor=True,
                                                  normalize_embeddings=True).to(self.device)

        self.candidates_embeddings = candidates_embeddings

        if output_directory_candidates_dump:
            # store candidate as pickle numpy array
            np.save(output_directory_candidates_dump + "candidates.npy", candidates_embeddings.cpu().numpy(),
                    allow_pickle=True)
            # save candidates as pickle list
            with open(output_directory_candidates_dump + 'candidates.pkl', 'wb') as f:
                pickle.dump(candidates, f)

            logger.info("Candidates encoded and stored in the output directory")

        else:
            logger.info("Candidates encoded")



    def load_candidates_from_files(self, output_directory_candidates_dump: str):
        """
        Load a faiss index from a given path
        :param faiss_index_path: path to the faiss index
        """

        candidates_embeddings = np.load(output_directory_candidates_dump + "candidates.npy", allow_pickle=True)

        self.candidates_embeddings = torch.from_numpy(candidates_embeddings).to(self.device)

        self.candidates = pickle.load(open(output_directory_candidates_dump + 'candidates.pkl', 'rb'))


    def retrieve_candidates(self, sequence_object: Sequence, top_k: int = 5):
        """
        This method retrieves the top k candidates for a given sequence_object
        :param sequence_object: sequence_object with the computed scores
        :param top_k: number of top candidates to retrieve
        :return:
        top_k_candidates: List of top k candidates
        top_k_scores: List of top k scores
        """

        if sequence_object.sum_computed_scores is None:
            raise ValueError("You need to model the sequence first")

        scores = sequence_object.sum_computed_scores

        top_k_scores, top_k_indices = torch.topk(scores, top_k)

        top_k_candidates = [self.candidates[i] for i in top_k_indices]

        return top_k_candidates, top_k_scores


    def update_sequence_with_computed_scores(self, sequence_object: Sequence, computed_scores: torch.Tensor):
        """
         updates the sequence_object with the new computed
        :param sequence_object:
        :param computed_scores: newly computed scores
        :return: updated sequence_object
        """
        if sequence_object.k_last_rows and not sequence_object.precompute:
            # get maximum utterance index in the context
            max_index = np.max([j for (i,j) in sequence_object.contextualized_embeddings_indices])

            # identify the last k indices in B2 space
            last_k_indices = list(range(max_index - sequence_object.k_last_rows, max_index))

            # use enumarate to get the index contain the last k indices
            all_last_k_indices = [i for i, tuple_value in enumerate(sequence_object.contextualized_embeddings_indices) if tuple_value[1] in last_k_indices]
            computed_scores = computed_scores[all_last_k_indices]
            sequence_object.computed_scores = computed_scores
            sequence_object.computed_scores_indices = [sequence_object.contextualized_embeddings_indices[i] for i in all_last_k_indices]
            sequence_object.sum_computed_scores = torch.sum(computed_scores, dim=0).detach().cpu()

        elif sequence_object.k_last_rows and sequence_object.precompute:
            # get maximum utterance index in precomputed scores
            combined_indices = sequence_object.computed_scores_indices + sequence_object.contextualized_embeddings_indices
            # extend precomputed scores with computed scores
            combined_scores = torch.cat([sequence_object.computed_scores, computed_scores], dim=0)
            # get maximum utterance index in the context
            max_index = np.max([j for (i,j) in combined_indices])

            # identify the last k indices in B2 space
            last_k_indices = list(range(max_index - sequence_object.k_last_rows, max_index))

            # get the index contain the last k indices
            all_last_k_indices = [i for i, tuple_value in enumerate(combined_indices) if tuple_value[1] in last_k_indices]
            sequence_object.computed_scores = combined_scores[all_last_k_indices]
            sequence_object.computed_scores_indices = [combined_indices[i] for i in all_last_k_indices]
            sequence_object.sum_computed_scores = torch.sum(computed_scores, dim=0).detach().cpu()

        elif sequence_object.precompute:
            sequence_object.sum_computed_scores = sequence_object.sum_computed_scores + torch.sum(computed_scores, dim=0).detach().cpu()
        else:
            sequence_object.sum_computed_scores = torch.sum(computed_scores, dim=0).detach().cpu()

        sequence_object.contextualized_embeddings = None
        sequence_object.contextualized_embeddings_indices = None
        sequence_object.precompute = True

        return sequence_object


    def sequence_modeling(self, sequence_object: Union[Sequence, List[Sequence]]):
        """
        This method computes the sequence for a single or a list of contexts
        :return: SequenceObject with the computed scores, if a single sequence object was passed
        or a list of sequence objects with the computed scores, if a list of sequence objects was passed
        """

        if self.candidates_embeddings is None:
            raise ValueError("You need to add candidates first")

        if isinstance(sequence_object, Sequence):
            computed_scores = torch.matmul(sequence_object.contextualized_embeddings, self.candidates_embeddings.T)
            sequence_object = self.update_sequence_with_computed_scores(sequence_object, computed_scores)

            return sequence_object

        all_contexts = []
        for sequence in sequence_object:
            all_contexts.append(sequence.contextualized_embeddings)

        all_contexts = torch.stack(all_contexts, dim=0)

        all_results = self._sequence_model_batch(all_contexts, evaluate=False)

        all_sequence_objects = []

        for i, sequence in enumerate(sequence_object):
            sequence = self.update_sequence_with_computed_scores(sequence, all_results[i])
            all_sequence_objects.append(sequence)

        return all_sequence_objects



    def contextualize_sequence(self,
                               context: List[str],
                               last_dialog_partner : Optional[bool] = False,
                               k_last_rows: Optional[int] = None,
                               ):
        """
        :param context: List of strings that will be used as context
        :param last_dialog_partner: If set to true, the last utterance in the context will be considered as even (useful if you want to precompute candidates)
        :param k_last_rows: if set, only the last k rows of the context (emerging triangle of states in triple encoders) will be used
        e.g. k_last_rows takes the last k utterances in the [B2] space and contexualize them with the entire [B1] space
        :return: SequenceObject with the context embeddings
        """

        context_b1 = ["[B1] [BEFORE] " + utterance for utterance in context]
        context_b2 = ["[B2] [BEFORE] " + utterance for utterance in context]

        encoded_context_b1 = self.model.encode(context_b1, show_progress_bar=False, convert_to_tensor=True).to(self.device)
        encoded_context_b2 = self.model.encode(context_b2, show_progress_bar=False, convert_to_tensor=True).to(self.device)

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
                                     ):
        """
        This method adds an utterance to the context and computes the sequence model.
        :param utterance: The utterance to add to the context
        :param sequence_object: The sequence object to which the utterance should be added.
        If None, a new sequence object will be created.
        :return: The updated sequence object
        """

        if sequence_object is None:
            sequence_object = Sequence()

        utterance_b1 = ["[B1] [BEFORE]  " + utterance]
        utterance_b2 = ["[B2] [BEFORE]  " + utterance]

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



    def _sequence_model_batch(self, context_embeddings_batch,
                              true_candidates_id: Optional[List[int]] = None,
                              batch_size: int = 32,
                              evaluate: bool = True):
        """
        This method computes the sequence model for a batch of context embeddings
        :param context_embeddings_batch: a batch of context embeddings
        :param true_candidates_id: a list of true candidates id for each dialogue to compute the rank (will be used if evaluate is True)
        :param batch_size: the batch size for the bmm across number of dialogues in parallel in the evaluator
        :param evaluate: if True, the ranks will be computed and returned, otherwise the bmm_result will be returned
        :return:
        """

        # now do for flattened_next_utterances_same_length_encoded
        next_utterances = self.candidates_embeddings

        # repeat next_utterances_all for each history in batch
        next_utterances_same_length_batch = next_utterances.repeat(batch_size, 1, 1).clone().detach().to(self.device)


        # calculate bmm history in batches and next_utterances_all
        for i in range(0, len(context_embeddings_batch), batch_size):
            if i + batch_size > len(context_embeddings_batch):
                history_tensor = context_embeddings_batch[i:len(context_embeddings_batch)].clone().detach().to(self.device).float()
                next_utterances_same_length_batch_slice = next_utterances.repeat(
                    len(context_embeddings_batch[i:len(context_embeddings_batch)]),
                    1, 1)
                if i == 0:
                    bmm_result = torch.bmm(next_utterances_same_length_batch_slice, history_tensor.transpose(1, 2))
                else:
                    bmm_result = torch.cat(
                        (
                            bmm_result,
                            torch.bmm(next_utterances_same_length_batch_slice, history_tensor.transpose(1, 2))),
                        dim=0)
            else:
                history_tensor = context_embeddings_batch[i:batch_size + i].clone().detach().to(self.device).float()
                if i == 0:
                    bmm_result = torch.bmm(next_utterances_same_length_batch, history_tensor.transpose(1, 2))
                else:
                    bmm_result = torch.cat(
                        (bmm_result, torch.bmm(next_utterances_same_length_batch, history_tensor.transpose(1, 2))),
                        dim=0)

        if not evaluate:
            return bmm_result

        # sum over the history dimension
        bmm_result = torch.sum(bmm_result, dim=2)

        ranks = []
        for his in range(len(context_embeddings_batch)):
            bm_entries = bmm_result[his]
            # flatten and sort
            bm_entries = bm_entries.flatten()
            # logging.info(f"bm_entries[next_utterances_id_same_length[his]]: {bm_entries[next_utterances_id_same_length[his]]}")

            true_utterance_score = bm_entries[true_candidates_id[his]]

            sorted_bm_entries = torch.sort(bm_entries, descending=True)

            # get rank of true utterance
            rank = torch.where(sorted_bm_entries[0] == true_utterance_score)[0][0] + 1
            ranks.append(rank.cpu())

        return ranks


    def evaluate_seq_dataset(self, dialogues : List[List[str]],
                             history_lengths: List[int] = [2,3,4,5,6,7,8,9,10],
                             last_k_rows: Optional[int] = None,
                             batch_size: int = 32):
        """
        This method evaluates the sequence model on a dataset of dialogues.
        :param dialogues: List of dialogues. Each dialogue is a list of utterances.
        :param history_lengths: List of history lengths to evaluate.
        :param last_k_rows: if set, only the last k rows of the context
        :param batch_size: Batch size for bmm across number of dialogues in parallel in the evaluator.
        """
        batch_size = min(batch_size, len(dialogues))

        report_dict = {}

        # flatten dialog text and create two arrays for dialog_id and utterance_id in dialog dialog_test
        flattened_dialog_test = [item for sublist in dialogues for item in sublist]

        # add after token to flatten dialog
        flattened_dialog_test_a = ["[AFTER]  " + utterance for utterance in flattened_dialog_test]

        flattened_dialog_test_encoded_a = self.model.encode(flattened_dialog_test_a, show_progress_bar=False,
                                                            normalize_embeddings=True)

        flattened_dialog_test_b1 = ["[B1] [BEFORE] " + utterance for utterance in flattened_dialog_test]

        flattened_dialog_test_b2 = ["[B2] [BEFORE] " + utterance for utterance in flattened_dialog_test]

        flattened_history_encoded_b_even_b1 = self.model.encode(flattened_dialog_test_b1,
                                                                convert_to_tensor=True, show_progress_bar=False)

        flattened_history_encoded_b_even_b2 = self.model.model.encode(flattened_dialog_test_b2,  convert_to_tensor=True,
                                                                      show_progress_bar=False)

        dialog_id = [i for i, sublist in enumerate(dialogues) for item in sublist]

        # reshape to dialog test based on dialog_id and utterance_id because not all dialogs have the same length
        flattened_dialog_test_encoded_reshaped = []
        flattened_dialog_test_encoded_reshaped_history_b1 = []
        flattened_dialog_test_encoded_reshaped_history_b2 = []
        for i, dialog in enumerate(dialogues):
            flattened_dialog_test_encoded_reshaped.append(
                flattened_dialog_test_encoded_a[dialog_id.index(i):dialog_id.index(i) + len(dialog)])
            flattened_dialog_test_encoded_reshaped_history_b1.append(
                flattened_history_encoded_b_even_b1[dialog_id.index(i):dialog_id.index(i) + len(dialog)])
            flattened_dialog_test_encoded_reshaped_history_b2.append(
                flattened_history_encoded_b_even_b2[dialog_id.index(i):dialog_id.index(i) + len(dialog)])

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
                    current_history_b1 = flattened_dialog_test_encoded_reshaped_history_b1[i][
                                              :history_length][::-1]
                    current_history_b2 = flattened_dialog_test_encoded_reshaped_history_b2[i][
                                              :history_length][::-1]

                    current_history_b1 = torch.tensor(current_history_b1)
                    current_history_b2 = torch.tensor(current_history_b2)

                    contextualized_embeddings, _ = self._contextualize_sequence(current_history_b1, current_history_b2, k_last_rows=last_k_rows)

                    history.append(contextualized_embeddings)

                    flattened_next_utterances_same_length.append(flattened_dialog_test_encoded_reshaped[i]
                                                                 [history_length])
                    next_utterances_id_same_length.append(iteration)
                    iteration += 1

                context_embeddings_batch = torch.nn.functional.normalize(torch.stack(history),
                                                                         p=2, dim=2).to(self.device)
                self.candidates_embeddings = torch.from_numpy(np.array(flattened_next_utterances_same_length)).to(self.device)

                ranks = self._sequence_model_batch(context_embeddings_batch, next_utterances_id_same_length, batch_size=batch_size)

                report_dict[f"H{history_length}_mean_rank"] = np.mean(ranks)

        df = pd.DataFrame(report_dict, index=[0])
        return df








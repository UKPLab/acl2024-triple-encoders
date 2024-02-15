import glob
import itertools
import os
import re
from typing import Optional, List
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from triple_encoders.TripleEncoders import TripleEncoders


class TripleEncodersForSTP(TripleEncoders):
    """
    Triple Encoders for Short-Term Planning in Dialogue (Goal-Oriented LLM Re-ranking)
    """

    def __init__(self,
                 model_name_or_path: Optional[str] = None,
                 llm_model_name_or_path = None
                 ):
        """
        :param model_name_or_path:
        :param speaker_token:
        :param llm_model_name_or_path: if you want to evaluate with a language model
        """
        if llm_model_name_or_path:
            self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name_or_path)
            self.llm_model.to(self.device)
            self.llm_model.eval()

        super().__init__(model_name_or_path)

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

        print(type(encoded_context_b1))
        print(len(encoded_context_b1))

        # reverse the order of the encoded context
        encoded_context_b2 = torch.flip(encoded_context_b2, (0,))
        encoded_context_b1 = torch.flip(encoded_context_b1, (0,))

        return encoded_context_b1, encoded_context_b2


    def compute_stp_scores(self, goal_embedding, candidates_embeddings, history):
        """
        Compute the scores for the Short-Term Planning approach
        :param goal_embedding: embedding of the goal to reach
        :param candidates_embeddings: candidate embeddings from the language model
        :param history: context embeddings
        :return:
        """
        # dot product between goal and all candidates
        scores_single = torch.matmul(torch.nn.functional.normalize(goal_embedding, p=2, dim=0),
                                     torch.nn.functional.normalize(candidates_embeddings, p=2, dim=0).T)


        # multiply history single scores by history length
        scores_single = scores_single * history.shape[0]

        # history dim (history_length, embedding_dim)
        # repeat history to match the length of combined utterances (history_length, embedding_dim) -> (combined_utterances.shape[0], history_length, embedding_dim)
        history = history.repeat(candidates_embeddings.shape[0], 1, 1)

        candidates_embeddings = candidates_embeddings.unsqueeze(1)  # Shape becomes [100, 1, 768]
        candidates_embeddings = candidates_embeddings.repeat(1, history.shape[1], 1)  # Shape becomes [100, 5, 768]

        # creat mean for every utterance in history and u_i in combined_utterances
        mean_history_candidate_utterances = (history + candidates_embeddings) / 2.0

        # Expand dimensions to perform batch-wise dot product
        goal_embedding_expanded = goal_embedding.unsqueeze(0).unsqueeze(0)  # Shape becomes [1, 1, 768]

        # Repeat the tensor to match the batch size
        goal_embedding_expanded = goal_embedding_expanded.repeat(mean_history_candidate_utterances.shape[0], 1,
                                                                 1)

        # normalize embeddings
        mean_history_candidate_utterances = torch.nn.functional.normalize(mean_history_candidate_utterances, p=2, dim=2)
        goal_embedding_expanded = torch.nn.functional.normalize(goal_embedding_expanded, p=2, dim=2)

        dot_products_single_goal = torch.bmm(mean_history_candidate_utterances,
                                             goal_embedding_expanded.transpose(1,
                                                                               2))  # Shape becomes [100, 5, 1]

        scores = torch.sum(dot_products_single_goal, dim=1)  # Shape becomes [100]

        scores_single = scores_single.unsqueeze(1)

        # scores = scores.squeeze(dim=1)

        scores = torch.cat((scores_single, scores), dim=1)

        scores = torch.sum(scores, dim=1)

        return scores

    def short_term_planning(self, candidates: List[str], goal: str, context: List[str], planning_odd: bool = True):
        """
        Short-Term Planning approach for ranking candidates to reach a goal
        :param candidates: List of candidate utterances
        :param goal: Goal utterance
        :param context: List of previous utterances (dialogue history)
        """
        if planning_odd:
            prefix = "[B2] [O] [BEFORE]  "
        else:
            prefix = "[B2] [E] [BEFORE]  "

        encode_context = self.encode_context_with_speaker_token(context, planning_odd)[0]

        candidates_with_prefix = [prefix + candidate for candidate in candidates]
        goal = "[AFTER] " + goal

        candidates_with_prefix.append(goal)

        embeddings \
            = self.model.encode(candidates_with_prefix, convert_to_tensor=True, normalize_embeddings=True).to(
            self.device)

        candidates_embeddings = embeddings[:-1]
        goal_embedding = embeddings[-1]

        scores = self.compute_stp_scores(goal_embedding,
                                         candidates_embeddings,
                                         encode_context)

        best_candidate_index = torch.argmax(scores).item()

        return candidates[best_candidate_index]


    def create_stp_dataset(self, dialogues: List[List[str]],
                           output_dir: str = 'data',
                           history_lengths: List[int] = [2, 5],
                           goal_distances: List[int] = [1,2,3,4]):
        """
        create a dataset for Short-Term Planning
        :param dialogues: list of dialogues
        :param output_dir: output directory
        :param history_lengths: list of history lengths
        :param goal_distances: list of goal distances

        """

        # create dir ltp_dataset
        if not os.path.exists(output_dir + '/stp_dataset'):
            os.makedirs(output_dir + '/stp_dataset')


        combinations = list(itertools.product(history_lengths, goal_distances))

        for history_length, goal_distance in combinations:
            histories = []
            candidates = []
            goals = []

            for i in tqdm(range(len(dialogues))):
                dialogue = dialogues[i]

                if len(dialogue) >= history_length + goal_distance + 1:

                    history = dialogue[:history_length]
                    candidate = dialogue[history_length]
                    goal = dialogue[history_length + goal_distance]

                    histories.append(history)
                    candidates.append(candidate)
                    goals.append(goal)

            df = pd.DataFrame({'History': histories, 'Candidate': candidates, 'Goal': goals})

            df['Candidate'] = df[f'Candidate'].astype(str)
            df['Goal'] = df[f'Goal'].astype(str)

            df.to_parquet(f'{output_dir}/stp_dataset/STP_H{str(history_length)}_D{str(goal_distance)}.parquet')




    def generate_llm_candidates(self, new_user_input_ids, top_p=0.9, num_return_sequences=100, temperature=0.8, **kwargs):
        chat_history_ids = self.llm_model.generate(new_user_input_ids,
                                                   max_length=1000,
                                                   do_sample=True,
                                                   pad_token_id=self.tokenizer.eos_token_id,
                                                   top_p=top_p,
                                                   num_return_sequences=num_return_sequences,
                                                   temperature=temperature,
                                                    **kwargs,
                                                   ).cpu().numpy()
        llm_candidates = [self.tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][i],
                                            skip_special_tokens=True) for i in range(num_return_sequences)]

        return llm_candidates


    def add_transformer_candidates(self, data_dir, top_p=0.8, num_return_sequences=100, temperature=0.8, **kwargs):
        """
        Add transformer candidates to the dataset

        :param data_dir: directory of the dataset
        :param top_p: nucleus sampling parameter
        :param num_return_sequences: number of candidates to generate
        :param temperature: temperature for LLM generation
        :param kwargs: other parameters for the generation
        :return:
        """
        if not self.llm_model:
            raise ValueError('LLM model was not provided in the constructor')

        files = glob.glob(data_dir + '/stp_dataset/*.parquet')

        # create dir for results
        if not os.path.exists(data_dir + '/stp_dataset_candidates'):
            os.makedirs(data_dir + '/stp_dataset_candidates')

        for file in tqdm(files):
            df = pd.read_parquet(file)
            df['GPT_candidates'] = None

            for index, row in tqdm(df.iterrows(), total=len(df)):
                history_list = row['History']
                history = ''
                for i in range(len(history_list)):
                    history += history_list[i] + self.tokenizer.eos_token

                input_ids =  self.tokenizer.encode(history, return_tensors='pt')

                gpt_candidates = self.generate_llm_candidates(input_ids, top_p, num_return_sequences, temperature, **kwargs)
                df.at[index, 'GPT_candidates'] = gpt_candidates

            file_name = os.path.basename(file)

            df.to_parquet(data_dir + f'/stp_dataset_candidates/{file_name}_P{temperature}_T{temperature}_N{num_return_sequences}.parquet')


    def evaluate_stp_dataset(self, data_dir = 'data'):
        """
        Evaluate the created STP dataset
        :param data_dir: directory of the dataset
        :return:
        """
        files = glob.glob(data_dir + '/stp_dataset_candidates/*.parquet')

        if len(files) == 0:
            raise ValueError('No files found in stp_dataset_candidates. Did you run add_transformer_candidates first?')

        # create dir for results
        if not os.path.exists(data_dir + '/stp_results'):
            os.makedirs(data_dir + '/stp_results')

        top_5s = []
        top_10s = []
        top_25s = []
        top_50s = []
        average_ranks = []
        history_lengths = []
        goal_distances = []
        for file in tqdm(files):
            df = pd.read_parquet(file)
            df['rank'] = None
            goal_distance = int(re.search(r'(?<=_D)\d+', file).group(0))
            history_length = int(re.search(r'(?<=_H)\d+', file).group(0))


            for index, row in tqdm(df.iterrows(), total=len(df)):
                history = row['History'][::-1]
                history_with_prefix = []
                if goal_distance % 2 == 0:
                    utterances = ["[B2] [E] [BEFORE]  " + utterance + " " for utterance in row['GPT_candidates']]
                    true_utterance = "[B2] [E] [BEFORE] " + row['Candidate']
                    for u_i in range(len(history)):
                        if u_i % 2 == 0:
                            history_with_prefix.append("[B1] [O] [BEFORE] " + history[u_i])
                        else:
                            history_with_prefix.append("[B1] [E] [BEFORE] " + history[u_i])
                else:
                    utterances = ["[B2] [O] [BEFORE]  " + utterance + " " for utterance in row['GPT_candidates']]
                    true_utterance = "[B2] [O] [BEFORE] " + row['Candidate']
                    for u_i in range(len(history)):
                        if u_i % 2 == 0:
                            history_with_prefix.append("[B1] [E] [BEFORE] " + history[u_i])
                        else:
                            history_with_prefix.append("[B1] [O] [BEFORE] " + history[u_i])

                goal_utterance = "[AFTER]  " + row['Goal']

                combined_utterances = utterances + [true_utterance] + [goal_utterance]
                combined_utterances = self.model.encode(combined_utterances, convert_to_tensor=True)
                history = self.model.encode(history_with_prefix, convert_to_tensor=True)

                goal_embedding = combined_utterances[-1]
                candidates_embeddings = combined_utterances[:-1]

                scores = self.compute_stp_scores(goal_embedding, candidates_embeddings, history)

                true_utterance_score = scores[-1]

                sorted_scores = scores.sort(descending=True).values

                df.at[index, 'rank'] = (sorted_scores.cpu().numpy() == true_utterance_score.cpu().numpy()).argmax() + 1

            df.to_parquet(data_dir + f'/stp_results/{os.path.basename(file)}')
            average_ranks.append(df['rank'].mean())
            top_5s.append(df[df['rank'] <= 5].shape[0] / df.shape[0])
            top_10s.append(df[df['rank'] <= 10].shape[0] / df.shape[0])
            top_25s.append(df[df['rank'] <= 25].shape[0] / df.shape[0])
            top_50s.append(df[df['rank'] <= 50].shape[0] / df.shape[0])
            history_lengths.append(history_length)
            goal_distances.append(goal_distance)

        df = pd.DataFrame({"History Length": history_lengths, "Goal Distance": goal_distances, 'Top 5': top_5s,
                           'Top 10': top_10s, 'Top 25': top_25s, 'Top 50': top_50s, 'Average Rank': average_ranks})

        df.to_csv(f'{data_dir}/stp_results/STP_Triple_results.csv', index=False)


        return df

















import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from ..SentenceTransformer import SentenceTransformer


class CosineSimilarityTripleEncoderLoss(nn.Module):
    """

    CosineSimilarityTripleEncoderLoss expects, that the InputExamples consists of three texts and a float label.

    It computes the vectors u' = model(input_text[0]), u'' = model(input_text[1]) and v = model(input_text[2]) and measures the cosine-similarity between the mean(u', u'') and v.
    By default, it minimizes the following loss: ||input_label - cos_score_transformation(cosine_sim(mean(u', u''),v))||_2.

    :param model: SentenceTranformer model
    :param loss_fct: Which pytorch loss function should be used to compare the cosine_similartiy(u,v) with the input_label? By default, MSE:  ||input_label - cosine_sim(u,v)||_2
    :param cos_score_transformation: The cos_score_transformation function is applied on top of cosine_similarity. By default, the identify function is used (i.e. no change).

    Example::

            from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses

            model = SentenceTransformer('TripleEncoder')
            train_examples = [InputExample(texts=['My first sentence', 'My second sentence', 'My third sentence'], label=0.8),
                InputExample(texts=['Another triple', 'related sentence', 'Unrelated sentence'], label=0.3)]
            train_dataset = SentencesDataset(train_examples, model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
            train_loss = losses.CosineSimilarityTripleEncoderLoss(model=model)

    """

    def __init__(self, model: SentenceTransformer, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity()):
        super(CosineSimilarityTripleEncoderLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_mean = torch.mean(torch.stack([embeddings[0], embeddings[1]]), dim=0)
        output = self.cos_score_transformation(torch.cosine_similarity(embeddings_mean, embeddings[2]))
        return self.loss_fct(output, labels.view(-1))


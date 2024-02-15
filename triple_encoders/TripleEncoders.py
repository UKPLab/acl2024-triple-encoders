import logging
from typing import Optional
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TripleEncoders:
    """
    TripleEncoders is a wrapper for the SentenceTransformer model.
    It is used to encode three texts and compute the cosine similarity between the mean of the first two texts and the third text.
    """
    def __init__(self, model_name_or_path: Optional[str] = None):
        """
        :param model_name_or_path: Path to the model or the model name from huggingface.co/models
        """
        self.model = SentenceTransformer(model_name_or_path)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = self.model.to(self.device)


































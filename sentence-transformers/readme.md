# Modifying Sentence-Transformers for training Triple Encoders

If you want to train your own triple encoders, make sure to install the sentence-transformers library. 
Once you have installed the library, add the following files to the respective directories in the sentence-transformers library.

- /losses/CosineSimilarityTripleEncoderLoss.py (This file contains the loss function for the triple encoder)
- /evaluators/EmbeddingsTripleSimilarityEvaluator.py (This file contains the evaluator for the triple encoder during training)

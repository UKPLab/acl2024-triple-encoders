<!--- BADGES: START, copied from sentence transformers, will be replaced with the actual once (removed for anonymity)--->
[![GitHub - License](https://img.shields.io/github/license/UKPLab/arxiv2024-triple-encoders?logo=github&style=flat&color=green)][#github-license]
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/triple-encoders?logo=pypi&style=flat&color=blue)][#pypi-package]
[![PyPI - Package Version](https://img.shields.io/pypi/v/triple-encoders?logo=pypi&style=flat&color=orange)][#pypi-package]


[#github-license]: https://github.com/
[#pypi-package]: https://pypi.org/project/triple-encoders/

<p align="center">
  <img align="center" src="img/triple-encoder-logo_with_border.png" width="430px" />
</p>

<p align="center">
    🤗 <a href="https://huggingface.co/UKPLab/triple-encoders-dailydialog" target="_blank">Models</a>  | 📃 <a href="https://aclanthology.org/2024.acl-long.290/" target="_blank">Paper</a>
</p>

`triple-encoders` is a library for contextualizing distributed [Sentence Transformers](https://sbert.net/) representations. 
At inference, triple encoders can be used for retrieval-based sequence modeling via sequential modular late-interaction: 
<p align="center">
  <img align="center" src="img/triple-encoder.jpg" width="1000px" />
</p>

Representations are encoded **separately** and the contextualization is **weightless**:
1. *mean-pooling* to pairwise contextualize sentence representations (creates a distributed query)
2. *cosine similarity* to measure the similarity between all query vectors and the retrieval candidates.
3. *summation* to aggregate the similarity (similar to average-based late interaction of [ColBERT](https://github.com/stanford-futuredata/ColBERT)).

## Key Features
- 1️⃣ **One dense vector vs distributed dense vectors**: in our paper we demonstrate that our late interaction-based approach outperforms single-vector representations on long sequences, including zero-shot settings.
- 🏎️💨 **Relative compute**: as every representation is encoded separately, you only need to encode, compute mixtures and similarities for the latest added representation (in dialog: the latest utterance). 
- 📚 **No Limit on context-length**: our distributed sentence transformer architecture is not limited to any sequence length. You can use your entire sequence as query!
- 🌎 **Multilingual support**: `triple-encoders` can be used with any [Sentence Transformers](https://sbert.net/) model. This means that you can model multilingual sequences by simply training on a multilingual model checkpoint. 


## Installation
You can install `triple-encoders` via pip:
```bash
pip install triple-encoders
``` 
Note that `triple-encoders` requires Python 3.6 or higher.

# Getting Started

Our experiments for sequence modeling and short-term planning conducted in the paper can be found in the `notebooks` folder. The hyperparameter that we used for training are the default parameters in the `trainer.py` file.

## Retrieval-based Sequence Modeling
We provide an example of how to use triple-encoders for conversational sequence modeling (response selection) with 2 dialog speakers. If you want to use triple-encoders for other sequence modeling tasks, you can use the `TripleEncodersForSequenceModeling` class.

### Loading the model
```python
from triple_encoders.TripleEncodersForConversationalSequenceModeling import TripleEncodersForConversationalSequenceModeling

# 🤗 model
triple_path = 'UKPLab/triple-encoders-dailydialog'

# load model
model = TripleEncodersForConversationalSequenceModeling(triple_path)
```

### Inference

```python
# load candidates for response selection 
candidates = ['I am doing great too!','Where did you go?', 'ACL is an interesting conference']

# load candidates and store index
model.load_candidates_from_strings(candidates, output_directory_candidates_dump='output/path/to/save/candidates')

# create a sequence
sequence = model.contextualize_sequence(["Hi!",'Hey, how are you?'], k_last_rows=2)

# model sequence (compute scores for candidates)
sequence = model.sequence_modeling(sequence)

# retrieve utterance from dialog partner
new_utterance = "I'm fine, thanks. How are you?"

# pass it to the model with dialog_partner=True 
sequence = model.contextualize_utterance(new_utterance, sequence, dialog_partner=True)

# model sequence (compute scores for candidates)
sequence = model.sequence_modeling(sequence)

# retrieve candidates to provide a response
response = model.retrieve_candidates(sequence, 3)
response
#(['I am doing great too!','Where did you go?', 'ACL is an interesting conference'],
# tensor([0.4944, 0.2392, 0.0483]))
```
**Speed:** 
- Time to load candidates: 31.815 ms
- Time to contextualize sequence: 18.078 ms
- Time to model sequence: 0.256 ms
- Time to contextualize new utterance: 15.858 ms
- Time to model new utterance: 0.213 ms
- Time to retrieve candidates: 0.093 ms

### Evaluation
```python
from datasets import load_dataset

dataset = load_dataset("daily_dialog")
test = dataset['test']['dialog']

df = model.evaluate_seq_dataset(test, k_last_rows=2)
df
# pandas dataframe with the average rank for each history length
```

## Short-Term Planning (STP)
Short-term planning enables you to re-rank candidate replies from LLMs to reach a goal utterance over multiple turns.

### Inference

```python
from triple_encoders.TripleEncodersForSTP import TripleEncodersForSTP

model = TripleEncodersForSTP(triple_path)

context = ['Hey, how are you ?',
           'I am good, how about you ?',
           'I am good too.']

candidates = ['Want to eat something out ?',
              'Want to go for a walk ?']

goal = ' I am hungry.'

result = model.short_term_planning(candidates, goal, context)

result
# 'Want to eat something out ?'
```
### Evaluation

```python
from datasets import load_dataset
from triple_encoders.TripleEncodersForSTP import TripleEncodersForSTP

dataset = load_dataset("daily_dialog")
test = dataset['test']['dialog']

model = TripleEncodersForSTP(triple_path, llm_model_name_or_path='your favorite large language model')

df = model.evaluate_stp_dataset(test)
# pandas dataframe with the average rank and Hits@k for each history length, goal_distance
```


# Training Triple Encoders
You can train your own triple encoders with Contextualized Curved Contrastive Learning (C3L) using our trainer. 
The hyperparameters that we used for training are the default parameters in the `trainer.py` file. 
Note that we pre-trained our best model with Curved Contrastive Learning (CCL) (from [imaginaryNLP](https://github.com/Justus-Jonas/imaginaryNLP)) before training with C3L.

```python
from triple_encoders.trainer import TripleEncoderTrainer
from datasets import load_dataset

dataset = load_dataset("daily_dialog")

trainer = TripleEncoderTrainer(base_model_name_or_path=,
                               batch_size=48,
                               observation_window=5,
                               speaker_token=True, # used for conversational sequence modeling
                               num_epochs=3,
                               warmup_steps=10000)

trainer.generate_datasets(
    dataset["train"]["dialog"],
    dataset["validation"]["dialog"],
    dataset["test"]["dialog"],
)


trainer.train("output/path/to/save/model")
```
## Citation
If you use triple-encoders in your research, please cite the following paper:
```
@inproceedings{erker-etal-2024-triple,
    title = "Triple-Encoders: Representations That Fire Together, Wire Together",
    author = "Erker, Justus-Jonas  and
      Mai, Florian  and
      Reimers, Nils  and
      Spanakis, Gerasimos  and
      Gurevych, Iryna",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.290",
    pages = "5317--5332",
    abstract = "Search-based dialog models typically re-encode the dialog history at every turn, incurring high cost.Curved Contrastive Learning, a representation learning method that encodes relative distances between utterances into the embedding space via a bi-encoder, has recently shown promising results for dialog modeling at far superior efficiency.While high efficiency is achieved through independently encoding utterances, this ignores the importance of contextualization. To overcome this issue, this study introduces triple-encoders, which efficiently compute distributed utterance mixtures from these independently encoded utterances through a novel hebbian inspired co-occurrence learning objective in a self-organizing manner, without using any weights, i.e., merely through local interactions. Empirically, we find that triple-encoders lead to a substantial improvement over bi-encoders, and even to better zero-shot generalization than single-vector representation models without requiring re-encoding. Our code (https://github.com/UKPLab/acl2024-triple-encoders) and model (https://huggingface.co/UKPLab/triple-encoders-dailydialog) are publicly available.",
}

```
# Contact
Contact person: Justus-Jonas Erker, justus-jonas.erker@tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.
This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

# License
triple-encoders is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.


### Acknowledgement
this package is based upon the [imaginaryNLP](https://github.com/Justus-Jonas/imaginaryNLP) and [Sentence Transformers](https://sbert.net/).




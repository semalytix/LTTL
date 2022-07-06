LTTL: Language and task informed transfer learning
==============

This code is based on the implementation of BLSE described in: 

Jeremy Barnes, Roman Klinger, and Sabine Schulde im Walde (2018): [**Bilingual Sentiment Embeddings: Joint Projection of Sentiment Across Languages**](http://aclweb.org/anthology/P18-1231). In Proceedings of ACL 2018.

For more details about LTTL please consult:

Katharina Allgaier, Susana Veríssimo, SherryTan, Matthias Orlikowski, & Matthias Hartung (2021): [**LLOD-driven Bilingual Word Embeddings Rivaling Cross-lingual Transformers in Quality of Life Concept Detection from French Online Health Communities**](https://doi.org/10.5281/zenodo.5011771). SEMANTiCS 2021, Amsterdam, The Netherlands.

Jorge Gracia, Christian Fäth, Matthias Hartung, Max Ionov, Julia Bosque-Gil, Susana Veríssimo, Christian Chiarcos, & Matthias Orlikowski (2020): [**LLeveraging Linguistic Linked Data for Cross-Lingual Model Transfer in the Pharmaceutical Domain**](https://doi.org/10.1007/978-3-030-62466-8_31). The Semantic Web – ISWC 2020 19th International Semantic Web Conference, Part II, 499–514.

and

Matthias Hartung, Matthias Orlikowski, Susana Veríssimo (2020): [**Evaluating the impact of bilingual lexical resources on cross-lingual sentiment projection in the pharmaceutical domain**](https://doi.org/10.5281/zenodo.3707940). Technical Report. 

Requirements
--------

The systems has been tested for python 3.6 and the requirements are set accordingly.


Usage
--------

Clone the repo:

```
git clone https://github.com/Semalytix/LTTL
```
As input the LTTL system needs:
* An annotated source and target language data set for a 2/3/4 category classification task (in .json format - see examples in datasets/)
* Embeddings for the source and target language (see examples in embeddings/)
* A lexicon that maps the vocabulary of the source language to the target language
* A configuration file that specifies the components above as well as (optionally) several hyper parameters and the choice of task (LTTL alone or in combination with another system (BL)) (in .yaml format - see detailed description and examples in configs/)

Run the code with the following command. All details will be specified in the configuration file.
```
python3 lttl.py -cf [path_to_config]
``` 




License
-------

Licensed under the terms of the [**Creative Commons CC-BY-NC public license**](https://creativecommons.org/licenses/)

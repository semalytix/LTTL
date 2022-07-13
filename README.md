LTTL: Language and task informed transfer learning
==============

This code is based on the implementation of BLSE described in: 

Jeremy Barnes, Roman Klinger, and Sabine Schulde im Walde (2018): [**Bilingual Sentiment Embeddings: Joint Projection of Sentiment Across Languages**](http://aclweb.org/anthology/P18-1231). In Proceedings of ACL 2018.

For more details about LTTL please consult:

Katharina Allgaier, Susana Veríssimo, SherryTan, Matthias Orlikowski, & Matthias Hartung (2021): [**LLOD-driven Bilingual Word Embeddings Rivaling Cross-lingual Transformers in Quality of Life Concept Detection from French Online Health Communities**](https://doi.org/10.5281/zenodo.5011771). SEMANTiCS 2021, Amsterdam, The Netherlands.

Jorge Gracia, Christian Fäth, Matthias Hartung, Max Ionov, Julia Bosque-Gil, Susana Veríssimo, Christian Chiarcos, & Matthias Orlikowski (2020): [**Leveraging Linguistic Linked Data for Cross-Lingual Model Transfer in the Pharmaceutical Domain**](https://doi.org/10.1007/978-3-030-62466-8_31). The Semantic Web – ISWC 2020 19th International Semantic Web Conference, Part II, 499–514.

Matthias Hartung, Matthias Orlikowski, Susana Veríssimo (2020): [**Evaluating the impact of bilingual lexical resources on cross-lingual sentiment projection in the pharmaceutical domain**](https://doi.org/10.5281/zenodo.3707940). Technical Report. 

Requirements
--------

The system has been tested for python 3.6 and the requirements are set accordingly. 


Usage
--------

Clone the repo:

```
git clone https://github.com/Semalytix/LTTL
```

Create a virtual environment and install the requirements via:

```
pip install -r requirements.txt
```

As input the LTTL system needs:
* An annotated source and target language data set for a 2/3/4 category classification task (in .json format - see examples in datasets/)
* Embeddings for the source and target languages
* A lexicon that maps the vocabulary of the source language to the target language (see examples in lexicons/)
* A configuration file that specifies the components above as well as (optionally) several hyper parameters and the choice of task (LTTL alone or in combination with another system (BL)) (in .yaml format - see detailed description and examples in configs/)

Run the code with the following command. All details will be specified in the configuration file.
```
python3 lttl.py -cf [path_to_config]
``` 


**Lexicon processing**

The folder lexicon_processing provides scripts to convert lexicons into the format required for LTTL, process it using different filters and to generate new lexicons via lexicon induction. Additionally, we provide the possibility to inspect and compare different lexica (for more information check `utils/lex_utils.py`)

If the lexicon that needs to be processed is already in .txt format or .tsv format, where the words on the left belong to one language and those on the right to another, the script `lex_creator.py` can be used.

This script requires a configuration file (.yaml) containing information about the type of lexicon to be created (simple, extended, inducted) and processing (basic, disambiguated, filtered). Depending on these criteria, further parameters need to be given to create the lexicon. For a template and example of a lexicon configuration, check `/lexicon_processing/lex_configs/`.

However, if the files are in other formats, a preliminary processing needs to be done. Here, we provide the following scripts to convert APERTIUM, MEDGLOSSARIES and MUSE lexicon source into LTTL-readable lexica:
* `process_apertium_lexicon.sh`
* `process_medglossaries.sh`
* `process_muse_lexica.sh`

An explanation and procedure for these lexical resources is provided in `PROCESSING_LEXICON_SOURCES.md`. 


License
-------

Licensed under the terms of the [**Creative Commons CC-BY-NC public license**](https://creativecommons.org/licenses/)

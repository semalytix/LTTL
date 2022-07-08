# Processing Lexicon Sources

The scripts in the folder `lex_sources_processing` allow for the creation of processing of bilingual lexicons that can be used within the LTTL framework. Included are processing methods for the following lexicon sources:
- Apertium
- Medglossaries
- Muse

The following sections describe the procedure used to obtain lexica from these lexicon sources. Although some preliminary processing can be done using these scripts, basic processing, lexicon disambiguation, lexicon filtered, lexicon extension and lexicon induction are all achieved using the script `lex_creator.py` and a configuration file (for examples check `\lex_configs`).

## Processing Apertium lexica

### Description

Currently, there are Apertium lexica for 22 language pairs. However, not all language-pairs of interest for the project have been created yet. These can be created via lexicon induction using one or more pivot languages. For that, the lexica with the format `src --> pivot` and `pivot --> trg` need to be obtained from the Apertium RDF graph. This can be achieved with this script: ```process_apertium_lexicon.sh```


### Procedure

Follow the procedure to create a bilingual dictionary in .tsv-format:

- download dictionaries of interest from [this repository](https://github.com/acoli-repo/acoli-dicts/tree/master/stable/apertium/apertium-rdf-2020-03-18) 
and save them to the folder ```data```. 

Please note that when you download the .zip files contained lexica in .ttl format, 
three lexica are available: a lexicon for the source language, a lexicon for the target language and a bilingual lexicon. 
The script uses the bilingual lexicon only, which is selected automatically; there is no need to delete the other lexica.
    
- run the script ```process_apertium_lexicon.sh``` from the terminal and select relevant functions to process the data:

    * ```process_apertium_lexicon.sh unzip_files``` will unzip either .zip or .gz files inside the data directory
    
    * ```process_apertium_lexicon.sh extract_simple_dictionary_ttl``` will parse the translation lexicon with .ttl format
    and retrieve a .tsv version, where the source language words are in the first column and the target language words 
    in the second column 
    
    * ```process_apertium_lexicon.sh extract_simple_dictionary_tsv``` will parse the bilingual lexicon provided in .tsv format
    and retrieve a new .tsv version, where the source language words are in the first column and the target language words 
    in the second column
    
    * ```process_apertium_lexicon.sh extract_same_pos_dic_entries_ttl``` will parse the translation in .ttl format and select
    entries that contain the same PoS in both source and target language and retrieve a .tsv lexicon where the source 
    language words are in the first column and the target language words in the second column 
    
    * ```process_apertium_lexicon.sh extract_same_pos_dic_entries_tsv``` will parse the bilingual lexicon provided in .tsv format
    and select entries that contain the same PoS in both source and target language and retrieve a new .tsv lexicon where 
    the source language entries are in the first column and the target word entries in the second column
    
    > **WARNING**: Not all words in the source language have a PoS assigned to them. The same applies to target language
    words. This means that the correspondence may not work out and you may not get the results you expect! It is safer to
     use ```extract_same_pos_dic_entries_ttl```. Refer to the code for more information.
    
    * ```process_apertium_lexicon.sh invert_tsv``` will invert the order of the columns from the processed .tsv. This function
    receives the path of the .tsv processed file as an argument. Use this function, if you want to invert the order of the 
    entries. For example, if your processed lexicon is fr-es.tsv (first function argument) and you want to have es-fr.tsv
    (second function argument), run: ```process_apertium_lexicon.sh invert_tsv fr-es.tsv es-fr.tsv```. Use either absolute or
    relative paths; if using relative paths mind that the working directory is data already.
    
The resulting lexica are in .tsv format. For further processing use the script `lex_creator.py` together with a config.

## Processing Medglossaries

### Description

Hand-crafted glossaries are a particularly valuable resource for the medical translator community and have shown to boost performance of MT systems. The MeSpEn_Glossaries: Medical Glossaries repository available here contains forty-six bilingual
medical glossaries for various language pairs generated from free online medical glossaries and dictionaries made by professional translators. Glossaries are encoded in standard tab-separated values (tsv) format. Because these glossaries 
cannot be readily used within the LTTL framework, a processing pipeline has to be created to transform the glossaries into valid LTTL lexicon files (.txt format). 

For this task the script ```process_medglossaries.sh``` can be used. The procedure to process the lexica is explained below.

### Procedure

Follow the procedure to create a bilingual dictionary in .txt-format:

- Download dictionaries of interest from [this resource](https://zenodo.org/record/2205690#.YEICuxBKiEs) 
and save them to the folder ``glossaries``.
    
- Run the script ```process_medglossaries.sh``` from the terminal and select relevant functions to process the data:

    * ```process_medglossaries.sh processing_pipeline``` will process tsv lexica into txt lexica, which are ready to use 
    in the LTTL framework (recommended).
    
    * ```process_apertium_lexicon.sh reformat_tsv_txt``` will reformat the lexica in tsv and transform them into lexica in 
    txt. This step also removes meta-information provided in the tsv (number of translations), which should not be in the 
    end lexicon (step 1).
    
    * ```process_apertium_lexicon.sh join_mwe``` will join multi-word expressions, i.e., expressions separated by spaces by 
    removing them. This step is necessary because the LTTL framework cannot process entries containing these (step 2).
    
After running the script with either of these functions, the txt lexica are available under ``medical_txt_lexica``. Files that have been processed using both steps end in "_nospaces.txt". For further processing use the script `lex_creator.py` together with a config.

## Processing MUSE lexica

### Description

MUSE is a Python library for multilingual word embeddings that provides SOTA multilingual unsupervised and supervised embeddings as well as large-scale high-quality bilingual dictionaries for training and evaluation.

In this cases, the interest lies on the bilingual lexica, which can be used in LTTL.
In total there are 110 large-scale ground-truth bilingual dictionaries that were created using a translation tool. These handle well the polysemy of words, according to the authors.

For this task the script ```process_muse_lexica.sh``` can be used. The procedure to process the lexica is explained below.

### Procedure:

- Download lexica of interest from [this resource](https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries) and save them to a folder `all_languages/`

- Run the script ```process_muse_lexica.sh```

The resulting lexica are in .txt format. For further processing use the script `lex_creator.py` together with a config.

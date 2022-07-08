#!/usr/bin/env bash

# In total there are 110 lexica. They were downloaded from here:
# https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries

# For easier download use wget https://dl.fbaipublicfiles.com/arrival/dictionaries/LANGUAGE-PAIR.txt
# (e.g. https://dl.fbaipublicfiles.com/arrival/dictionaries/en-fr.txt)

# All lexica were saved into a folder all_languages/

# Gets the paths of all txt files saved previously in these folders
lexica=$(ls all_languages/*.txt)

# Iterates over the files in the folder "all_languages" and processes them to make them compatible with the lttl
# framework, i.e., source language and target language words are tab separated, multi-word expressions are turned into
# single word expressions by removing spaces in-between and entries are sorted and de-duplicated

for file in $lexica;
   do
  	echo $file
  	lexname=$(basename "$file"|cut -f 1 -d ".")
    lexname+="_processed.txt"
  	echo $lexname
  	awk '{print $1 "\t" $2}' $file|sed 's/ //g'|sort|uniq > ${lexname} ;
  	mv ${lexname} ./all_languages
   done

# Warning: if the script is ran more than once, files that end in "_processed" will not be deleted, this suffix is repeated!

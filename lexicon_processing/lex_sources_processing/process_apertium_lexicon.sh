#!/usr/bin/env bash

# Script to process and format Apertium lexica with .ttl and .tsv formats to be used in the LTTL framework

# The tsv output is a file with two columns
#### col 1: entries in the source language
#### col 2: entries in the target language

#Apertium lexica are available here: https://github.com/acoli-repo/acoli-dicts/tree/master/stable/apertium/apertium-rdf-2020-03-18

mkdir -p tsv_lexica

#Unzip files from either .zip or .gz
unzip_files(){
zipfiles=`ls *.zip *.gz`
for file in $zipfiles;
    do
        if [[ $file == *.zip ]]; then  unzip -n $file
        elif [[ $file == *.gz ]]; then  gzip -d $file
        echo $file; fi;
    done

}

### Lexicon extraction for .ttl format (Recommended)

# Extract and process a simple dictionary from a TranslationSet*.ttl file
extract_simple_dictionary_ttl() {
filenames=`ls *TranslationSet*.ttl`

for eachfile in $filenames;
    do
        language_pair=$(echo $eachfile | cut -d'_' -f1)

        echo $language_pair

        # Extract and format each bilingual lexicon. Delete entries without a translation
        output=$(less $eachfile | grep "vartrans:trans :" | awk -F" " '{print $3}' |awk -F"-" '{print $1}' | sed s/://g | awk -F"_" '{if ($2 != "") print $1"\t"$2}'|sed -e 's/%/\\\x/g')

        #Format (url decoder) and sort entries and eliminate duplicates
        printf '%b' "$output"|sort|uniq > ${language_pair}_sorted_unique.tsv

        mv **_sorted_unique.tsv ./tsv_lexica
    done
}

# Extract and process a dictionary from a TranslationSet*.ttl file and select only the entries that have the save PoS in
# the source and target language
extract_same_pos_dic_entries_ttl(){

filenames=`ls *TranslationSet*.ttl`

for eachfile in $filenames;
    do
        src_lang=$(echo $eachfile | cut -d'_' -f1| cut -d '-' -f2)
        trg_lang=$(echo $eachfile | cut -d'_' -f1| cut -d '-' -f3)

        echo $src_lang
        echo $trg_lang
        echo $eachfile

        # Extract source language PoS Tags
        src_pos=$(less $eachfile | grep "vartrans:source :" | awk -F ":" '{print $3}'|  awk -v src=${src_lang} '{split($0, a, "-"src"-sense;"); print a[1]}'|awk -F\- '{print $NF}')

        # Extract target language PoS Tags
        trg_pos=$(less $eachfile | grep "vartrans:target :" | awk -F ":" '{print $3}'|  awk -v trg=${trg_lang} '{split($0, a, "-"trg"-sense ."); print a[1]}'|awk -F\- '{print $NF}')

        # Extract source language words
        src_w=$(less $eachfile | grep "vartrans:trans :" | awk -F" " '{print $3}' |awk -F"-" '{print $1}' | sed s/://g | awk -F"_" '{if ($2 != "") print $1}'|sed -e 's/%/\\\x/g')
        #Format source language words (url decoder)
        src_w_format=$(printf '%b' "$src_w/n")

        # Extract target language words
        trg_w=$(less $eachfile | grep "vartrans:trans :" | awk -F" " '{print $3}' |awk -F"-" '{print $1}' | sed s/://g | awk -F"_" '{if ($2 != "") print $2}'|sed -e 's/%/\\\x/g')

        #Format target language words (url decoder)
        trg_w_format=$(printf '%b' "$trg_w/n")

        # Check if the src_lang POS = trg_lang POS, if yes save respective src_lang - trg_lang entry in csv (which is sorted
        # and duplicates are eliminated. First empty line is also eliminated)
        paste <(printf '%s' "$src_w_format") <(printf '%s' "$src_pos") <(printf '%s' "$trg_w_format") <(printf '%s' "$trg_pos") | awk -F "\t" '{if ($2 == $4) print $1"\t"$3}'|sort|uniq| awk NF > ${src_lang}_${trg_lang}_ttl_pos_sort_unique.tsv

        mv **ttl_pos_sort_unique.tsv ./tsv_lexica
    done
}


### Lexicon extraction for tsv format (Warning:PoS processing may be faulty, depending on the format of the tsv files!)

# Extract and process a dictionary from a trans*.tsv file
extract_simple_dictionary_tsv(){
filenames=`ls trans*.tsv`

for eachfile in $filenames;
    do
        language_pair=$(echo $eachfile |  cut -d'_' -f2|cut -d '.' -f1)

        echo $language_pair

        # Extract and format each bilingual lexicon. Delete entries without a translation
        output=$(less $eachfile | grep "tranSet"| awk '{print $3}' |awk -F"/" '{print $7}' | awk -F"-" '{print $1}' | awk -F"_" '{if ($2 != "") print $1"\t"$2}'|sed -e 's/%/\\\x/g')

        #Format (url decoder) and sort entries and eliminate duplicates
        printf '%b' "$output"|sort|uniq > ${language_pair}_sorted_unique.tsv

        mv **_sorted_unique.tsv ./tsv_lexica
    done
}

# Extract and process a dictionary from a trans*.tsv file and select only the entries that have the save PoS in the source and target language
extract_same_pos_dic_entries_tsv(){
filenames=`ls trans*.tsv`

for eachfile in $filenames;
    do
        src_lang=$(echo $eachfile |  cut -d'_' -f2|cut -d '.' -f1|cut -d "-" -f1)
        trg_lang=$(echo $eachfile |  cut -d'_' -f2|cut -d '.' -f1|cut -d "-" -f2)

        echo $src_lang
        echo $trg_lang

        # Not all URIs contain pos-tag information. This retrieve the last part of the URI before the language code
        src_pos=$(less $eachfile |  awk '{print $2}' |awk -F"/" '{print $7}'| awk -v src=${src_lang} '{split($0,a, "-"src">"); print a[1]}'| awk -F\- '{print $NF}')

        # Not all URIs contain pos-tag information. This retrieve the last part of the URI before the language code
        trg_pos=$(less $eachfile | awk '{print $6}' | awk -F"/" '{print $7}' | awk -v trg=${trg_lang} '{split($0,a, "-"trg">"); print a[1]}'| awk -F\- '{print $NF}')

        #Extract source language words
        src_w=$(less $eachfile |  awk '{print $1}' |awk -F "@" '{print $1}' |sed -e 's/^"//' -e 's/"$//')

        # Extract target language words
        trg_w=$(less $eachfile |  awk '{print $NF}' |awk -F "@" '{print $1}' |sed -e 's/^"//' -e 's/"$//')

        # Check if the src_lang POS = trg_lang POS, if yes save respective src_lang - trg_lang entries in csv (which is sorted and duplicates are eliminated)
        # Only those words that have matching pos are part of the dictionary.
        # If a word does not have pos information, it will be excluded from this dictionary
        paste <(printf '%s' "$src_w") <(printf '%s' "$src_pos") <(printf '%s' "$trg_w") <(printf '%s' "$trg_pos") | awk -F "\t" '{if ($2 == $4) print $1"\t"$3}'|sort|uniq > ${src_lang}_${trg_lang}_tsv_pos_sort_unique.tsv

        mv **_tsv_pos_sort_unique.tsv ./tsv_lexica
    done
}


#Invert produced tsv files to make sure that src is in col 1 and trg in col 2
# Note that the working directory is specified already, mind this if inserting relative path
invert_tsv(){

    awk '{print $2 "\t" $1}' $1 > $2
}


### Calling function from the command-line ####

# Check if the function exists (bash specific)
if declare -f "$1" > /dev/null
then
  # call arguments verbatim
  "$@"
else
  # Show a helpful error
  echo "'$1' is not a known function name" >&2
  exit 1
fi

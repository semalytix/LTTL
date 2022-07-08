#!/usr/bin/env bash

# The medglossaries can be downloaded with this link: https://zenodo.org/record/2205690#.Ysc4--xBzep
# The MeSpEn_Glossaries collection contains forty-six bilingual medical glossaries for various language pairs generated
# from free online medical glossaries and dictionaries made by professional translators.

mkdir -p tsv_lexica

### Reformat tsv medglossaries to txt format

# Creates a bilingual lexicon in txt format from a glossary in tsv format
reformat_tsv_txt(){

    # Copy your files to a folder named glossaries. The created files should end in medglossaries.tsv
    filenames=$(ls glossaries/*medglossaries.tsv)

    for eachfile in $filenames;
        do
            # Create the name for the new .txt lexicon
            lexname=$(basename "$eachfile"|cut -f 1 -d ".")
            lexname+=".txt"

            # Remove the number and space from the first column. This number indicates the number of meanings a word has
            # Create a .txt file where entries are separated by tabs
            sed 's/^.. //g' $eachfile | awk -F"\t" '{print $1 "\t" $2}'|sort|uniq > ${lexname}

            #Move the files to the folder medical_txt_lexica
            mv ${lexname} ./medical_txt_lexica
        done

}

### Process multi-word expressions (MWE)

# Joins multi-word expressions in the txt lexicon to form single words so that the lexicon can be used in the LTTL framework
join_mwe(){

    # All files are saved under the folder glossaries (in data) and the files end in medglossaries.txt
    filenames=$(ls medical_txt_lexica/*medglossaries.txt)
    for eachfile in $filenames;

        do
            # Create the name for the new .txt lexicon
           lexname=$(basename "$eachfile"|cut -f 1 -d ".")
           lexname+="_nospaces.txt"

           #Remove spaces from entries containing multi-word expressions (MWE)
           sed 's/ //g' $eachfile > ${lexname}

            #Move the files to the folder medical_txt_lexica
           mv ${lexname} ./medical_txt_lexica

        done
}

### Processing pipeline

# 2-step processing pipeline to convert tsv medglossaries into txt medglossaries for use in LTTL
processing_pipeline(){

    echo "Starting processing pipeline."
    echo "Reformating tsv lexica into txt lexica..."
    reformat_tsv_txt
    echo "Joining multi-word expressions..."
    join_mwe
    echo "The processing is now complete!"

}

## Calling function from the command-line ####

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

#!/usr/bin/env bash

#cd data/

mkdir -p converted_mwe_swe

### Processing of multi-word expressions (MWE)

# Processes .txt lexica containing MWE by removing the spaces between them and saving them as single-word expressions (SWE)
# One script for processing of files in a directory - join_mwe_folder and one script for processing of single files - join_mwe_file

join_mwe_folder(){
    # Read in the directory path and saves it into a variable
    read -p 'Enter the directory path: ' directory

    for eachfile in $directory/*;
        do
           # Creates the name for each new lexicon
           lexname=$(basename "$eachfile"|cut -f 1 -d ".")
           lexname+="_nospaces.txt"

           # Converts MWE into SWE
           sed 's/ //g' $eachfile > ${lexname}

           # Moves converted lexicon to folder ./converted_mwe_swe
           mv ${lexname} ./converted_mwe_swe

        done
}

join_mwe_file(){

   # Reads in the file path and saves it into a variable
   read -p 'Enter the file path: ' file

   # Creates the name for the new lexicon
   lexname=$(basename "$file"|cut -f 1 -d ".")
   lexname+="_nospaces.txt"

   # Converts MWE into SWE
   sed 's/ //g' $file > ${lexname}

   # Moves converted lexicon to folder ./converted_mwe_swe
   mv ${lexname} ./converted_mwe_swe

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

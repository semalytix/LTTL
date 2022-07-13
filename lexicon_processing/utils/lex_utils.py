import copy
from collections import Counter, defaultdict
import csv
import os


class BasicLexicon:
    """
    Creates an object from a txt or tsv lexicon.

    The lexicon object can be used by itself, to get the entries from the bilingual lexicon and transform them into a
    dictionary object and be further processed to remove duplicate entries and multi-word expressions (via joining them)
    for use in the LTTL framework.

    BasicLexicon is used as a basis for more complex lexicon classes.
    ----------
    Attributes
        read_path: path from txt or tsv file of the lexicon of interest
        save_path: path to save lexicon processed with the methods of this class. Default=None
        basic_processing: boolean, default=True. Transforms multi-word expressions into single words and
        de-duplicates entries.
    """

    def __init__(self, read_path, save_path=None, basic_processing=True):
        self.read_path = read_path
        self.save_path = save_path
        self.basic_processing = basic_processing
        self.lexicon = defaultdict(list)

        # Load file to self.lexicon, file must be in txt or tsv format
        if self.read_path.endswith("txt"):
            self.open_lexicon_txt()
        elif self.read_path.endswith("tsv"):
            self.open_lexicon_tsv()
        else:
            raise Exception(f"input file '{self.read_path}' must be txt or tsv format.")

        # Checks if basic processing is true or false
        if self.basic_processing:
            # Run basic processing pipeline for all entries in the lexicon
            print(f"Running processing pipeline for lexicon in {self.read_path}")
            self.lexicon = self.basic_processing_pipeline(self.lexicon)

        # If a saving path is given, the lexicon can be saved into a txt file
        if self.save_path:
            print(f"Saving lexicon to {self.save_path}")
            self.save_txt_lexicon(self.lexicon)

    def open_lexicon_txt(self):
        """
        Opens an existing txt bilingual lexicon file
        :return: object lexicon populated with source lang words and their corresponding translations
        """

        with open(self.read_path, "r") as f:
            data = f.read().splitlines()
            entries = [entry.strip().split('\t') for entry in data]
            # This solves the problem with entries containing "'t"
            for i, item in enumerate(entries):
                if len(item) != 2:
                    entries[i] = item[0].split(" ")
        for src, trg in entries:
            self.lexicon[src].append(trg)

    def open_lexicon_tsv(self):
        """
        Opens an existing tsv bilingual lexicon file.

        If there are more than two columns in the file, only the first two are considered. It is important that the first
        column of the file contains source language words and the second column the corresponding translation per word.

        :return: object lexicon populated with source lang words and their corresponding translations
        """

        with open(self.read_path) as f:
            data = csv.reader(f, delimiter="\t")
            for row in data:
                if row[0] not in self.lexicon.keys():
                    self.lexicon[row[0]] = []  # create entry with empty list
                translations = self.lexicon[row[0]]
                translations.append(row[1])

    def save_txt_lexicon(self, lexicon):
        """
        Creates a txt file for the processed lexicon.

        :param lexicon: The lexicon object or defaultdict(list)
        :return: A txt file with the following format: src_lang entry "\t" target_lang entry
        """

        with open(self.save_path, 'w') as f:
            for src_entry, trg_entry in sorted(lexicon.items()):
                for e in trg_entry:
                    entry = src_entry + '\t' + e + '\n'
                    f.writelines(entry)

    def transform_mwe_sw(self, lexicon):
        """
        Transforms entries in the lexicon that have multiple-word expressions in single words by removing spaces between
        words.

        This is done so that the final, processed lexicon can be read by the lttl framework, as it cannot handle
        multi-word expressions
        :param lexicon: lexicon object or defaultdict(list)
        :return: a lexicon where the entries can only consist of a word
        """

        for src_entry in [k for k in lexicon.keys()]:
            if " " in src_entry:
                new_src_entry = "".join(src_entry.split())
                lexicon[new_src_entry] = lexicon[src_entry]
                del lexicon[src_entry]
        for trg_entries in [v for v in lexicon.values()]:
            # Iterate over list of translations per key
            for index, e in enumerate(trg_entries):
                if " " in e:
                    # replace entry with multi-word expression for a single word expression (via space removal)
                    new_e = "".join(e.split())
                    trg_entries[index] = new_e
        return lexicon

    def duplicate_remover(self, lexicon):
        """
        Removes duplicate entries from the dictionary.

        :param lexicon: lexicon object or defaultdict(list)
        :return: a lexicon without duplicate translations
        """
        for src_entries, trg_entries in lexicon.items():
            lexicon[src_entries] = list(set(trg_entries))
        return lexicon

    def basic_processing_pipeline(self, lexicon):
        """
        Creates a lexicon without multi-word expressions and duplicate entries.

        :param lexicon: lexicon object or defaultdict(list)
        :return: final lexicon
        """
        # Run transformation of multi-word expressions into single words for source lang and target lang words
        lexicon = self.transform_mwe_sw(lexicon)

        # Run deduplication to remove duplicated target words
        lexicon = self.duplicate_remover(lexicon)

        return lexicon


class Corpus:
    """
    Creates a corpus object, which can be used for the disambiguation of lexicon entries.

    In addition to the corpus, the number of total tokens is saved. A Corpus object is used in the classes
    DisambiguatedLexicon and FilteredLexicon, but it can be used for other purposes.
    ----------
    Attributes:
        corpus_path: path for the corpus os interest.
    """

    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.tokens = []

        # Creates a list of words contained in the corpus saved to self.tokens
        self.create_corpus()

        # Counts the number of tokens in a corpus for statistics purposes
        self.number_tokens = len(set(self.tokens))

    def create_corpus(self):
        """
        Reads in a corpus and gets its tokens in lowercase
        :return: list of tokens
        """
        with open(self.corpus_path, "r") as f:
            data = f.read().splitlines()
            tokens_per_sent = [sentence.split(" ") for sentence in data]
            self.tokens = [item.lower() for sublist in tokens_per_sent for item in sublist]


class DisambiguatedLexicon(BasicLexicon):
    """
    Creates a disambiguated lexicon using an existing lexicon and a corpus.

    The corpus in the target language is used to disambiguate the lexicon, so that when there are multiple translations,
    those that are the most frequent in the corpus are saved in the lexicon. The least frequent translations are discarded.
    However, if there are translations in the lexicon that do not match any word in the corpus, these are not removed

    The purpose of this class is to create a lexicon with disambiguated translations, in case of multiple word senses.

    This class inherits the methods from BasicLexicon. The basic_processing parameter is as set to True as default.
    ----------
    Attributes
        read_path: path from txt or tsv file of the lexicon of interest
        save_path: path to save lexicon processed with the methods of this class. Default=None
        disambiguation_corpus_path: path to a disambiguation corpus
    """

    def __init__(self, read_path, save_path, disambiguation_corpus_path):
        super().__init__(read_path, save_path, basic_processing=True)

        self.disambiguation_corpus_path = disambiguation_corpus_path
        self.disambiguation_corpus = Corpus(disambiguation_corpus_path)

        self.lexicon = self.disambiguation_pipeline(self.lexicon, self.disambiguation_corpus.tokens)
        print(f"Lexicon {self.read_path} was disambiguated using corpus {self.disambiguation_corpus_path}.")

        if self.save_path:
            print(f"Saving lexicon to {self.save_path}")
            self.save_txt_lexicon(self.lexicon)

    def matched_translations(self, lexicon, corpus):
        """
        Creates a dictionary where the keys are the matches between the disambiguation corpus and lexicon and the values
        are the frequency of the words in the corpus
        :param lexicon: defaultdict
        :param corpus: corpus object
        :return: matches between lexicon and corpus according to frequency (Counter object)
        """
        matches = {}
        corpus_word_frequency = Counter(corpus)
        for word, freq in corpus_word_frequency.items():
            for translation_candidates in lexicon.values():
                if word in translation_candidates:
                    matches[word] = freq
        return matches

    def disambiguation_pipeline(self, lexicon, corpus):
        """
        Creates a lexicon that selects entries according to the frequency of translations in a corpus, if more than one
        translation is found in the disambiguation corpus.

        Translations that are not matched by the corpus, but present in the lexicon are still part of the final lexicon.
        :param lexicon: defaultdict(list)
        :param corpus: a Corpus object
        :return: disambiguated lexicon
        """
        # Create the matches dictionary
        matched_translation_candidates = self.matched_translations(lexicon, corpus)

        # Create a copy of the lexicon being disambiguated
        disambiguated_lexicon = copy.deepcopy(lexicon)

        for k, v in disambiguated_lexicon.items():
            # checks if there is more than one translation available
            if len(v) > 1:
                # saves matches in a temporary list with tuples with the form (word frequency, word)
                tmp = [[matched_translation_candidates[word], word] for word in v if
                       word in matched_translation_candidates.keys()]
                if len(tmp) > 0:
                    # if there are matches available, the word with the highest frequency is chosen
                    highest_frequent_word = max(tmp)[1]
                    # The disambiguated lexicon keeps only the highest frequent matches words and nothing else for the key
                    # Because the disambiguated lexicon is a copy of a lexicon, all other translations not matched in
                    # the corpus are kept
                    disambiguated_lexicon[k] = [highest_frequent_word]
        return disambiguated_lexicon


class FilteredLexicon(BasicLexicon):
    """
    Creates a filtered lexicon using an existing lexicon and a corpus.

    The corpus in the target language is used to disambiguate the lexicon, so that when there are multiple translations,
    those that are the most frequent in the corpus are saved in the lexicon. The least frequent translations are discarded.
    Any other translations from the lexicon not matched in the corpus are discarded as well.

    The purpose of this class is to create a lexicon with disambiguated translations only.

    This class inherits the methods from BasicLexicon. The basic_processing parameter is set to True as default.
    ----------
    Attributes
        read_path: path from txt or tsv file of the lexicon of interest
        save_path: path to save lexicon processed with the methods of this class. Default=None
        disambiguation_corpus_path: path to a disambiguation corpus
    """

    def __init__(self, read_path, save_path, disambiguation_corpus_path):
        super().__init__(read_path, save_path, basic_processing=True)

        self.disambiguation_corpus_path = disambiguation_corpus_path
        self.disambiguation_corpus = Corpus(disambiguation_corpus_path)

        self.lexicon = self.filtering_pipeline(self.lexicon, self.disambiguation_corpus.tokens)
        print(f"Lexicon {self.read_path} was filtered using corpus {self.disambiguation_corpus_path}.")

        if self.save_path:
            print(f"Saving lexicon to {self.save_path}")
            self.save_txt_lexicon(self.lexicon)

    def matched_translations(self, lexicon, corpus):
        """
        Creates a dictionary where the keys are the matches between the disambiguation corpus and lexicon and the values
        are the frequency of the words in the corpus
        :param lexicon: defaultdict
        :param corpus: corpus object
        :return: matches between lexicon and corpus according to frequency (Counter object)
        """

        matches = {}
        corpus_word_frequency = Counter(corpus)
        for word, freq in corpus_word_frequency.items():
            for translation_candidates in lexicon.values():
                if word in translation_candidates:
                    matches[word] = freq
        return matches

    def filtering_pipeline(self, lexicon, corpus):
        """
        Creates a lexicon that selects entries according to the frequency of translations in a corpus, if more than one
        translation is found in the disambiguation corpus.

        Translations that are not matched by the corpus are not added to the filtered lexicon

        :param lexicon: lexicon object or default dict
        :param corpus: corpus object
        :return: filtered lexicon
        """
        # Create the matches dictionary
        # Translations that are found in the corpus are added to the dictionary and counted according to the frequency
        # in the corpus
        matched_translation_candidates = self.matched_translations(lexicon, corpus)

        filtered_lexicon = {}
        for k, v in lexicon.items():
            # checks, if there is more than one translation available
            if len(v) > 1:
                # saves matches in a temporary list with tuples with the form (word frequency, word)
                tmp = [[matched_translation_candidates[word], word] for word in v if
                       word in matched_translation_candidates.keys()]
                if len(tmp) > 0:
                    # if there are matches available, the word with the highest frequency is chosen
                    highest_frequent_word = max(tmp)[1]
                    # The filtered lexicon keeps only the highest frequent matches words and nothing else for the key
                    filtered_lexicon[k] = [highest_frequent_word]

        return filtered_lexicon


class ExtensionLexicon:
    """
    Creates an extension lexicon based on a baseline lexicon and additional lexica.

    Any lexicon can be used as a baseline lexicon. Additional lexica are given as a list.

    This class uses methods from BasicLexicon, DisambiguatedLexicon and FilteredLexicon depending on the processing chosen
    for the lexica involved. In all cases, basic_processing is set to True.
    ----------
    Attributes
        baseline_lex_path: path from txt or tsv file of the baseline lexicon
        additional_lex_paths: a list of paths of lexica used for extension of the baseline lexicon
        corpus_path: path to a disambiguation corpus if disambiguation or filtering is desired
        save_path: path to save lexicon processed with the methods of this class. Default=None
        filter_method: default="basic" (same as BasicLexicon)
    """

    def __init__(self, baseline_lex_path, save_path, additional_lex_paths: list, corpus_path=None,
                 filter_method="basic"):
        self.baseline_lex_path = baseline_lex_path
        self.save_path = save_path
        if corpus_path:
            self.corpus_path = corpus_path
        self.additional_lex_paths = additional_lex_paths
        self.extension_lexicon = None  # defined in self.extension_lexicon_pipeline()

        # Creates appropriate lexicon objects using the paths depending on filtering
        if filter_method == "basic":
            self.baseline = BasicLexicon(baseline_lex_path, None)
            self.additional_lexicons = [BasicLexicon(lex, None) for lex in additional_lex_paths]

        elif filter_method == "disambiguation":
            self.baseline = DisambiguatedLexicon(baseline_lex_path, None, self.corpus_path)
            self.additional_lexicons = [DisambiguatedLexicon(lex, None, self.corpus_path) for lex in
                                        additional_lex_paths]

        elif filter_method == "filtering":
            self.baseline = FilteredLexicon(baseline_lex_path, None, self.corpus_path)
            self.additional_lexicons = [FilteredLexicon(lex, None, self.corpus_path) for lex in additional_lex_paths]

        # Runs extension pipeline using the baseline lexicon object and additional lexicon objects
        self.extension_lexicon_pipeline(self.baseline, self.additional_lexicons)

        # Saves files into a txt if a path for saving (save_path) is provided
        if self.save_path:
            print(f"Saving lexicon to {self.save_path}")
            self.save_txt_lexicon(self.extension_lexicon.lexicon)

    @staticmethod
    def extend_lexicon(baseline_lexicon, additional_lexicon):
        """
        Extends a baseline lexicon with one additional lexicon.

        Any lexicon can be used as a baseline lexicon. The baseline lexicon is read in first and then the additional lexicon.
        If for a source language word there is a new translation from the additional lexicon, the translation will be
        inserted in the list of translations for that word. If the source word does not exist in the baseline lexicon, the entry
        will be added to the lexicon.
        :param baseline_lexicon: a lexicon object
        :param additional_lexicon: a lexicon object
        :return: a default dict
        """

        #Create a copy of the baseline lexicon
        extended = copy.deepcopy(baseline_lexicon)
        # Check if the src lang word is available in the additional lexicon
        for src_entry, trg_entry in additional_lexicon.lexicon.items():
            if src_entry in extended.lexicon.keys():
                for e in trg_entry:
                    #Add a translation to the list of translations for the key
                    extended.lexicon[src_entry].insert(0, e)
            else:
                # Add a new entry to the lexicon (src, trg)
                extended.lexicon[src_entry] = trg_entry
        return extended

    def extension_lexicon_pipeline(self, baseline_lexicon, additional_lexicons: list):
        """
        The pipeline allows for the extension of a lexicon using multiple additional lexicons instead of one.

        Any lexicon can be used as a baseline lexicon. The additional lexicons are loaded from a list of lexicon objects.
        The pipeline applies the extend_lexicon method to all additional lexicons; the final lexicon contains all changes

        :param baseline_lexicon: lexicon object
        :param additional_lexicons: list with lexicon objects
        :return: extended lexicon
        """

        self.extension_lexicon = baseline_lexicon
        print(f"Baseline lexicon is {baseline_lexicon.read_path}")
        for lex in additional_lexicons:
            print(f"Running extension pipeline with lexicon {lex.read_path}")
            self.extension_lexicon = self.extend_lexicon(self.extension_lexicon, lex)
        print(f"Extension pipeline ran successfully! In total {len(additional_lexicons) + 1} lexicons were used.")

    def save_txt_lexicon(self, lexicon):
        """
        Creates a txt file for the processed lexicon.

        :param lexicon: The lexicon object or defaultdict(list)
        :return: A txt file with the following format: src_lang entry "\t" target_lang entry
        """
        with open(self.save_path, 'w') as f:
            for src_entry, trg_entry in sorted(lexicon.items()):
                for e in trg_entry:
                    entry = src_entry + '\t' + e + '\n'
                    f.writelines(entry)


class InductionLexicon:
    """
    Creates a lexicon for a source and target language pair using pivot languages.

    This method relies on having one or more pivot languages that have translations in the source and target language of
    interest. Two lexica are needed per pivot:
        - a lexicon with the source language and pivot language translations (source -> pivot)
        - a lexicon with the pivot language and target language translations (pivot -> target)
    The translation candidates are obtained via transitive closure (source - pivot - target). If there is a translation
    for the pivot word in the target language, the new lexicon will get the word from the source language and attribute the
    translation in the target language, where there is a common pivot word.

    This class uses methods from BasicLexicon, DisambiguatedLexicon and FilteredLexicon depending on the processing chosen
    for the lexica involved. In all cases, basic_processing is set to True.
    ----------
    Attributes
        source_lang: source language of interest (e.g. en)
        target_lang: target language of interest (e.g. fr)
        pivot_langs: dictionary with keys pivot_lang, path_with_source, path_with_target
        corpus_path: path to a disambiguation corpus if disambiguation or filtering is desired
        save_path: path to save lexicon processed with the methods of this class. Default=None
        filter_method: default="basic" (same as BasicLexicon)
    """

    def __init__(self, source_lang, target_lang, pivot_langs, corpus_path=None, save_path=None, filter_method="basic"):

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.pivot_langs = pivot_langs
        self.save_path = save_path
        self.filter_method = filter_method
        if corpus_path:
            self.corpus_path = corpus_path
            self.disambiguation_corpus = Corpus(self.corpus_path)
        self.final_lexicon = None  # defined in self.induction_pipeline(self.mapping)
        self.mapping = []  # needed in order to save inducted lexica from each pivot and creating the final lexicon below

        for pivot in self.pivot_langs:
            # Iterates over pivot_langs dict to create tuples of the form (pivot_lang, lex_with_src, lex_with_trg)
            pivot_lang, lex_with_src_path, lex_with_trg_path = pivot["pivot_lang"], pivot["lex_with_src"], pivot[
                "lex_with_trg"]
            self.pivot_lang = pivot_lang
            self.lex_with_src_path = lex_with_src_path
            self.lex_with_trg_path = lex_with_trg_path

            print(
                f"Loading lexica for induction for pivot {self.pivot_lang} : {self.lex_with_src_path}, {self.lex_with_trg_path}")
            self.lex_with_src = BasicLexicon(self.lex_with_src_path, None)
            self.lex_with_trg = BasicLexicon(self.lex_with_trg_path, None)
            print("Adding entries to induction lexicon")
            self.inducted_lexicon = self.induct_lexicon(self.lex_with_src, self.lex_with_trg)
            self.mapping.append(self.inducted_lexicon)

        # Runs induction pipeline using one or more pivot languages
        self.final_lexicon = self.induction_pipeline(self.mapping)
        print(f"Lexicon object for the language pair {self.source_lang}-{target_lang} was created.")

        # Applies filter methods to lexicon, if specified
        print(f"Processing lexicon entries with filter method: {self.filter_method}.")
        if self.filter_method == "disambiguation":
            self.final_lexicon = self.disambiguation_pipeline(self.final_lexicon, self.disambiguation_corpus.tokens)
            print(
                f"Lexicon for language pair {self.source_lang}-{self.target_lang} was disambiguated using {self.corpus_path}.")
        elif self.filter_method == "filtering":
            self.final_lexicon = self.filtering_pipeline(self.final_lexicon, self.disambiguation_corpus.tokens)
            print(
                f"Lexicon for language pair {self.source_lang}-{self.target_lang} was filtered using {self.corpus_path}.")

        # Saves files into a txt if a path for saving (save_path) is provided
        if self.save_path:
            print(f"Saving lexicon to {self.save_path}")
            self.save_txt_lexicon(self.final_lexicon)

    def induct_lexicon(self, lex_with_source, lex_with_target):
        """
        Creates a lexicon via transitive closure using one pivot language.

        :param lex_with_source: a lexicon object where the keys belong to the source language and values to the pivot language
        :param lex_with_target: a lexicon object where the keys belong to the pivot language and the values to the target language
        :return: inducted lexicon with keys in the source language and values in the target language
        """
        lexicon = {}
        # iterate over translation pairs (src, pivot) in the lexicon containing the source language of interest
        for src_key, pivot_target in lex_with_source.lexicon.items():
            # check if the pivot is part of the lexicon with the target language of interest
            for pivot_key in lex_with_target.lexicon.keys():
                # if the pivot in the lexicon with the target language is found in the lexicon with the source language
                if pivot_key in pivot_target:
                    if src_key not in lexicon:
                        # Add key in the source language
                        lexicon[src_key] = []  # create entry with empty list
                    # Add translations in the target language
                    lexicon[src_key].extend(lex_with_target.lexicon[pivot_key])
        return lexicon

    def induction_pipeline(self, mapping: list):
        """
        The pipeline allows for lexicon induction using multiple pivots.

        It gets each inducted lexicon via one pivot from the mapping list and creates a final lexicon.
        :param mapping: list with inducted lexica
        :return: final induction lexicon
        """

        # create final lexicon dictionary
        lexicon = {}

        # Create a de-duplicated set of src words
        words = []
        for dictionary in mapping:
            for key in dictionary.keys():
                words.append(key)
        words = set(words)

        # Populate lexicon with target lang translations
        for word in words:
            for dictionary in mapping:
                if word in dictionary.keys():
                    if word not in lexicon:
                        lexicon[word] = []
                    lexicon[word].extend(dictionary[word])  # flatten list of trg translations
            # de-duplicate translations
            lexicon[word] = list(set(lexicon[word]))
        return lexicon

    def matched_translations(self, lexicon, corpus):
        """
        Creates a dictionary where the keys are the matches between the disambiguation corpus and lexicon and the values
        are the frequency of the words in the corpus
        :param lexicon: dict
        :param corpus: corpus object
        :return: matches between lexicon and corpus according to frequency (Counter object)
        """
        matches = {}
        corpus_word_frequency = Counter(corpus)
        for word, freq in corpus_word_frequency.items():
            for translation_candidates in lexicon.values():
                if word in translation_candidates:
                    matches[word] = freq
        return matches

    def disambiguation_pipeline(self, lexicon, corpus):
        """
        Creates a lexicon that selects entries according to the frequency of translations in a corpus, if more than one
        translation is found in the disambiguation corpus.

        Translations that are not matched by the corpus, but present in the lexicon are still part of the final lexicon
        :param lexicon: defaultdict(list)
        :param corpus: a Corpus object
        :return: disambiguated lexicon
        """
        # Create the matches dictionary
        matched_translation_candidates = self.matched_translations(lexicon, corpus)

        # Create a copy of the lexicon being disambiguated
        disambiguated_lexicon = copy.deepcopy(lexicon)

        for k, v in disambiguated_lexicon.items():
            # checks if there is more than one translation available
            if len(v) > 1:
                # saves matches in a temporary list with tuples with the form (word frequency, word)
                tmp = [[matched_translation_candidates[word], word] for word in v if
                       word in matched_translation_candidates.keys()]
                if len(tmp) > 0:
                    # if there are matches available, the word with the highest frequency is chosen
                    highest_frequent_word = max(tmp)[1]
                    # The disambiguated lexicon keeps only the highest frequent matches words and nothing else for the key
                    # Because the disambiguated lexicon is a copy of a lexicon, all other translations not matched in
                    # the corpus are kept
                    disambiguated_lexicon[k] = [highest_frequent_word]
        return disambiguated_lexicon

    def filtering_pipeline(self, lexicon, corpus):
        """
        Creates a lexicon that selects entries according to the frequency of translations in a corpus, if more than one
        translation is found in the disambiguation corpus.

        Translations that are not matched by the corpus are not added to the filtered lexicon

        :param lexicon: lexicon object or default dict
        :param corpus: corpus object
        :return: filtered lexicon
        """
        # Create the matches dictionary
        # Translations that are found in the corpus are added to the dictionary and counted according to the frequency
        # in the corpus
        matched_translation_candidates = self.matched_translations(lexicon, corpus)

        filtered_lexicon = {}
        for k, v in lexicon.items():
            # checks, if there is more than one translation available
            if len(v) > 1:
                # saves matches in a temporary list with tuples with the form (word frequency, word)
                tmp = [[matched_translation_candidates[word], word] for word in v if
                       word in matched_translation_candidates.keys()]
                if len(tmp) > 0:
                    # if there are matches available, the word with the highest frequency is chosen
                    highest_frequent_word = max(tmp)[1]
                    # The filtered lexicon keeps only the highest frequent matches words and nothing else for the key
                    filtered_lexicon[k] = [highest_frequent_word]

        return filtered_lexicon

    def save_txt_lexicon(self, lexicon):
        """
        Creates a txt file for the processed lexicon.

        :param lexicon: The lexicon object or defaultdict(list)
        :return: A txt file with the following format: src_lang entry "\t" target_lang entry
        """
        with open(self.save_path, 'w') as f:
            for src_entry, trg_entry in sorted(lexicon.items()):
                for e in trg_entry:
                    entry = src_entry + '\t' + e + '\n'
                    f.writelines(entry)


class LexiconInspector:
    """
    This class compares one lexicon with another and provides information about their common entries, different entries
    and conflicts in their source language words and translations.

    The respective entries can be exported, if export=True

    This class uses methods from BasicLexicon with basic_processing set to True, because it is assumed that any
    filtering occurred in previous steps when creating the lexica.
    ----------
    Attributes
        lex1_path: path of the first lexicon
        lex2_path: path of the second lexicon
    """

    def __init__(self, lex1_path, lex2_path):
        self.lex1_path = lex1_path
        self.lex2_path = lex2_path

        # Transform the lexicons into simple dictionaries instead of defaultdict(list) to avoid hidden empty keys/values
        self.lex1 = dict(BasicLexicon(self.lex1_path).lexicon)
        self.lex2 = dict(BasicLexicon(self.lex2_path).lexicon)

        # Create list of source-target language translation pairs for each lexicon
        self.entries_lex1 = [(src, entry) for src, trg in self.lex1.items() for entry in trg]
        self.entries_lex2 = [(src, entry) for src, trg in self.lex2.items() for entry in trg]

        # Create a list of target-source language translation pairs for each lexicon ("reverse")
        self.entries_lex_1_reverse = [(entry, src) for src, trg in self.lex1.items() for entry in trg]
        self.entries_lex_2_reverse = [(entry, src) for src, trg in self.lex2.items() for entry in trg]

    def get_intersection(self, export=False):
        """
        Gets the intersection of entries between two lexica.

        The intersection is comprised of common language pairs with the format (src_lang, trg_lang) in both lexica.

        :param export: default=False
        :return: txt file, if export=True
        """

        # Get translation pairs of source language word and target language word for each lexicon
        intersection = [entry for entry in self.entries_lex2 if entry in self.entries_lex1]
        print(f"{self.lex1_path} and {self.lex2_path} have {len(intersection)} common entries")
        if export:
            dest = open(
                f"intersect_{os.path.splitext(os.path.basename(self.lex1_path))[0]}_{os.path.splitext(os.path.basename(self.lex2_path))[0]}.txt",
                "w+")
            for src, trg in intersection:
                dest.write(src + "\t" + trg + "\n")
            dest.close()

    def get_differences(self, export=False):
        """
        Gets the differences between entries between two lexica.

        For the language pairs with the format (src_lang, trg_lang) in both lexica, it is checked which ones are different
        and to which lexicon they belong. The differences per lexicon are saved to separate files/printed in the console,
        depending on the value of the param export.

        :param export: default=False
        :return: two txt files, one per lexicon, if export=True
        """

        # Get translation pairs that are only available in the first lexicon
        lex1_only = [entry for entry in self.entries_lex1 if entry not in self.entries_lex2]

        # Print number of missing translation in the second lexicon
        print(self.lex1_path, "has", len(lex1_only), "entries that are missing in", self.lex2_path)
        if export:
            # Export the missing translation pairs (missing from the second lexicon)
            dest = open(
                f"diff_{os.path.splitext(os.path.basename(self.lex1_path))[0]}>{os.path.splitext(os.path.basename(self.lex2_path))[0]}.txt",
                "w+")
            for src, trg in lex1_only:
                dest.write(src + "\t" + trg + "\n")
            dest.close()

        # Get translation pairs that are only available in the second lexicon
        lex2_only = [entry for entry in self.entries_lex2 if entry not in self.entries_lex1]

        # Print number of missing translation in the first lexicon
        print(self.lex2_path, "has", len(lex2_only), "entries that are missing in", self.lex1_path)

        if export:
            # Export the missing translation pairs (missing from the first lexicon)
            dest = open(
                f"diff_{os.path.splitext(os.path.basename(self.lex2_path))[0]}>{os.path.splitext(os.path.basename(self.lex1_path))[0]}.txt",
                "w+")
            for src, trg in lex2_only:
                dest.write(src + "\t" + trg + "\n")
            dest.close()

    def get_conflicts(self, export=False):
        """
        Gets the conflicting source language words and target language words between two lexica per word.

        For the language pairs with the format (src_lang, trg_lang) in both lexica it is checked:
            - If for a source lang word, there is a different conflicting translation between lex1 and lex2
            - If for a target lang word, there is a different source lang word correspondence between lex1 and lex2

        :param export: default=False
        :return: two txt files, one per lexicon, if export=True
        """

        # Save conflicts from source to target language word between lexica in a list
        src_trg_conflicts = []

        # Assess which target language words (translations) possess different keys, thus causing a conflict
        for entry in self.entries_lex1:
            for e in self.entries_lex2:
                if entry[0] == e[0] and entry[1] != e[1]:
                    src_trg_conflicts.append((entry[0], entry[1], e[1]))
        print(self.lex1_path, "and", self.lex2_path, "have", len(src_trg_conflicts),
              "conflicting translations (SRC-->TRG)")

        if export:
            # Export the conflicting source language words for a translation word in lexicon1 and lexicon2
            dest = open(
                f"conflicts_SRC_TRG_{os.path.splitext(os.path.basename(self.lex1_path))[0]}_{os.path.splitext(os.path.basename(self.lex2_path))[0]}.txt",
                "w+")
            for src, trg, e in src_trg_conflicts:
                dest.write(src + "\t" + trg + "\t" + e + "\n")
            dest.close()

        # Save conflicts from target to source language word between the two lexica in a list
        trg_src_conflicts = []

        # Assess which target language words (translations) possess different keys, thus causing a conflict
        for entry in self.entries_lex_1_reverse:
            for e in self.entries_lex_2_reverse:
                if entry[0] == e[0] and entry[1] != e[1]:
                    trg_src_conflicts.append((entry[0], entry[1], e[1]))

        # Print number of conflicts with respect to the target language words
        print(self.lex1_path, "and", self.lex2_path, "have", len(trg_src_conflicts),
              "conflicting translations (TRG-->SRC)")
        if export:
            # Export the conflicting translations and their correspondent source language words in lexicon1 and lexicon2
            dest = open(
                f"conflicts_TRG_SRC_{os.path.splitext(os.path.basename(self.lex1_path))[0]}_{os.path.splitext(os.path.basename(self.lex2_path))[0]}.txt",
                "w+")
            for trg, src, e in trg_src_conflicts:
                dest.write(trg + "\t" + src + "\t" + e + "\n")
            dest.close()

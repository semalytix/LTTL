import sys
from utils.lex_utils import *
import yaml


def config_reader(config_path):
    """
    Reads the lexicon configurations from the yaml file.

    :param config_path: path of configuration yaml file
    :return: config dict
    """
    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    return config


def config_parser(config):
    """
    Reads the configuration from the config dict.

    In the configuration it is possible to specify the configuration of more than one lexicon at a time and for different
    types of lexica. Running time depends on the number of lexica and processing options.

    :param config: config dict
    :return: lexicon object
    """
    for l in config:
        print(f"Loading lexicon configuration: {l['lexicon']}")
        type, processing, save_path = l['lexicon']["type"], l['lexicon']["processing"], l['lexicon']["save_file"]
        print(f"Lexicon type: {type}")
        print(f"Lexicon processing: {processing}")
        if type == "simple":
            if processing == "basic":
                lexicon = BasicLexicon(l["lexicon"]["baseline_lex"], save_path)
            elif processing == "disambiguation":
                lexicon = DisambiguatedLexicon(l["lexicon"]["baseline_lex"], save_path,
                                               l["lexicon"]["disambiguation_corpus"])
            elif processing == "filtering":
                lexicon = FilteredLexicon(l["lexicon"]["baseline_lex"], save_path,
                                          l["lexicon"]["disambiguation_corpus"])
        elif type == "extension":
            if processing == "basic":
                lexicon = ExtensionLexicon(l["lexicon"]["baseline_lex"], save_path,
                                           l["lexicon"]["additional_lex"])
            elif processing == "disambiguation":
                lexicon = ExtensionLexicon(l["lexicon"]["baseline_lex"], save_path,
                                           l["lexicon"]["additional_lex"],
                                           l["lexicon"]["disambiguation_corpus"],
                                           filter_method=processing)
            elif processing == "filtering":
                lexicon = ExtensionLexicon(l["lexicon"]["baseline_lex"], save_path,
                                           l["lexicon"]["additional_lex"],
                                           l["lexicon"]["disambiguation_corpus"],
                                           filter_method=processing)
        elif type == "induction":
            if processing == "basic":
                lexicon = InductionLexicon(l["lexicon"]["source_lang"], l["lexicon"]["target_lang"],
                                           l["lexicon"]["pivot_langs"], None, save_path)
            elif processing == "disambiguation":
                lexicon = InductionLexicon(l["lexicon"]["source_lang"], l["lexicon"]["target_lang"],
                                           l["lexicon"]["pivot_langs"],
                                           l["lexicon"]["disambiguation_corpus"], save_path,
                                           filter_method=processing)
            elif processing == "filtering":
                lexicon = InductionLexicon(l["lexicon"]["source_lang"], l["lexicon"]["target_lang"],
                                           l["lexicon"]["pivot_langs"],
                                           l["lexicon"]["disambiguation_corpus"], save_path,
                                           filter_method=processing)

    return lexicon


if __name__ == "__main__":
    lex_configuration = config_reader(sys.argv[1])
    print("Loading configuration file." + "\n" + f"configuration file path: {sys.argv[1]}" + "\n")
    print("Running lexicon pipeline")
    config_parser(lex_configuration)
    print("The configuration was run successfully!")

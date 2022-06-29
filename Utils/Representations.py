import numpy as np
from nltk import sent_tokenize,word_tokenize

def ave_vecs(sentence, model):
    """
    Returns an embedding which is the average
    of the word embeddings from the word embedding
    model for the tokens in a sentence.
    """
    sent = np.array(np.zeros((model.vector_size)))
    sent_length = len(sentence.split())
    for w in sentence.split():

        try:
            sent += model[w]
        except KeyError:
            sent += np.random.uniform(-.25, .25, model.vector_size)

    return sent / sent_length


# def words(sentence, model):
#     """
#     Returns a list of tokens using the nltk word_tokenizer. This is mainly
#     for the General_Dataset class.
#
#     """
#     # TODO still needed?
#     return word_tokenize(sentence)



def getMyData(list_sents, label, model, representation=ave_vecs, encoding='utf8'):
    """
    Returns a list of representations (either ave_vecs or words) and label tuples
    which serve as x, y pairs for training.
    """
    data = []
    for sent in list_sents:
        data.append((representation(sent, model), label))
    return data

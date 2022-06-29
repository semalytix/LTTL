import numpy as np

class WordVecs(object):
    """
    Helper class for importing word embeddings.
    These embeddings can be either in
    word2vec formats (txt or bin) or 
    glove format.

    If you already know the vocabulary you want to
    import, you can pass it as a set or list
    in the 'vocab' parameter.
    """


    def __init__(self, file, file_type='word2vec', vocab=None, encoding='utf8'):
        self.file_type = file_type
        self.vocab = vocab
        self.encoding = encoding
        (self.vocab_length, self.vector_size, self._matrix,
         self._w2idx, self._idx2w) = self._read_vecs(file)

    def __getitem__(self, y):
        try:
            return self._matrix[self._w2idx[y]]
        except KeyError:
            raise KeyError
        except IndexError:
            raise IndexError
        

    def _read_vecs(self, file):
        """Assumes that the first line of the file is
        the vocabulary length and vector dimension."""

        if self.file_type == 'word2vec':
            txt = open(file, encoding=self.encoding).readlines()
            vocab_length, vec_dim = [int(i) for i in txt[0].split()]
            txt = txt[1:]
        elif self.file_type == 'bin':
            txt = open(file, 'rb', encoding=self.encoding)
            header = txt.readline()
            vocab_length, vec_dim = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * vec_dim

        else:
            txt = open(file).readlines()
            vocab_length = len(txt)
            vec_dim = len(txt[0].split()[1:])


        if self.vocab:
            emb_matrix = np.zeros((len(self.vocab), vec_dim))
            vocab_length = len(self.vocab)
        else:
            emb_matrix = np.zeros((vocab_length, vec_dim))
        w2idx = {}

        # Read a binary file
        if self.file_type == 'bin':
            for line in range(vocab_length):
                word = []
                while True:
                    ch = txt.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                # if you have vocabulary, you can only load these words
                if self.vocab:
                    if word in self.vocab:
                        w2idx[word] = len(w2idx)
                        emb_matrix[w2idx[word]] = np.fromstring(txt.read(binary_len), dtype='float32')  
                    else:
                        txt.read(binary_len)
                else:
                    w2idx[word] = len(w2idx)
                    emb_matrix[w2idx[word]] = np.fromstring(txt.read(binary_len), dtype='float32')  

        # Read a txt file
        else:    
            for item in txt:
                if self.file_type == 'tang':            # tang separates with tabs
                    split = item.strip().replace(',','.').split()
                else:
                    split = item.strip().split(' ')
                try:
                    word, vec = split[0], np.array(split[1:], dtype=float)

                    # if you have vocabulary, only load these words
                    if self.vocab:
                        if word in self.vocab:
                            w2idx[word] = len(w2idx)
                            emb_matrix[w2idx[word]] = vec
                        else:
                            pass
                    else:
                        if len(vec) == vec_dim:
                            w2idx[word] = len(w2idx)
                            emb_matrix[w2idx[word]] = vec
                        else:
                            pass
                except ValueError:
                    pass

            
        idx2w = dict([(i, w) for w, i in w2idx.items()])

        return vocab_length, vec_dim, emb_matrix, w2idx, idx2w

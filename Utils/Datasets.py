import os
import random
from Utils.Representations import *

random.seed(42)

class ProjectionDataset():
    """
    A wrapper for the translation dictionary. The translation dictionary
    should be word to word translations separated by a tab. The
    projection dataset only includes the translations that are found
    in both the source and target vectors.
    """
    def __init__(self, translation_dictionary, src_vecs, trg_vecs):
        (self._Xtrain, self._Xdev, self._ytrain,
         self._ydev) = self.getdata(translation_dictionary, src_vecs, trg_vecs)
        self.info = self.pdataset_info(translation_dictionary)

    def train_dev_split(self, x, train=.9):
        # split data into training and development, keeping /train/ amount for training.
        train_idx = int(len(x)*train)
        return x[:train_idx], x[train_idx:]

    def getdata(self, translation_dictionary, src_vecs, trg_vecs):
        x, y = [], []
        with open(os.path.join(os.path.split(translation_dictionary)[0],"unmatched.txt"),
                  "w", encoding="utf-8") as file:
            with open(translation_dictionary, encoding="utf-8") as f:
                for line in f:
                    src, trg = line.split()
                    try:
                        _ = src_vecs[src]
                        _ = trg_vecs[trg]
                        x.append(src)
                        y.append(trg)
                    except KeyError:
                        file.write(line +"\n")
        xtr, xdev = self.train_dev_split(x)
        ytr, ydev = self.train_dev_split(y)
        return xtr, xdev, ytr, ydev

    def pdataset_info(self,translation_dictionary):
        with open(translation_dictionary,encoding="utf-8") as f:
            length = len([line for line in f])
            return length


class CorpusDataset(object):


    def __init__(self, fname, model,number_cat,one_hot=True, dtype=np.float32, rep=ave_vecs, lowercase=True,train_set=True, combination=False):
        self.rep = rep
        self.one_hot = one_hot
        self.lowercase = lowercase
        self.train_set = train_set
        self.combination = combination
        self.number_cat = number_cat

        self.dataset = self.open_json(fname)
        if number_cat == "binary":
            self.pos, self.neg = self.get_data(self.dataset)
            self.pos_text = self.shuffle_data(self.get_text(self.pos))
            self.neg_text = self.shuffle_data(self.get_text(self.neg))

            #added docID and shuffled
            self.pos_ids = self.shuffle_data(self.get_docid(self.pos))
            self.neg_ids = self.shuffle_data(self.get_docid(self.neg))

            self.pos_span, self.neg_span = self.get_span((self.pos, self.neg))
            if train_set:
                self._Xtrain, self._Xdev, self._Xtest, self._ytrain, self._ydev, self._ytest = self.build_dataset(number_cat=number_cat, model=model, representation=rep, train_set=train_set, combination=combination)
            else:
                if combination:
                    #add docID for target data only (so no train_set)
                    self._Xtest, self._ytest, self._docid_test = self.build_dataset(number_cat=number_cat, model=model, representation=rep, train_set=train_set, combination=combination)
                else:
                    #add docID for target data only (so no train_set)
                    self._Xdev, self._Xtest, self._ydev, self._ytest, self._docid_dev, self._docid_test = self.build_dataset(number_cat=number_cat ,model=model, representation=rep, train_set=train_set, combination=combination)
        elif number_cat == "three class":
            self.pos, self.neg, self.neu = self.get_data(self.dataset)
            self.pos_text = self.shuffle_data(self.get_text(self.pos))
            self.neg_text = self.shuffle_data(self.get_text(self.neg))
            self.neu_text = self.shuffle_data(self.get_text(self.neu))
            self.pos_span, self.neg_span, self.neu_span = self.get_span((self.pos, self.neg, self.neu))
            if train_set:
                self._Xtrain, self._Xdev, self._Xtest, self._ytrain, self._ydev, self._ytest = self.build_dataset(number_cat=number_cat,
                                                                                                                  model=model,
                                                                                                                  representation=rep,
                                                                                                                train_set=train_set, combination=combination)
            else:
                if combination:
                    self._Xtest, self._ytest = self.build_dataset(number_cat=number_cat, model=model,
                                                                  representation=rep, train_set=train_set,
                                                                  combination=combination)
                else:
                    self._Xdev, self._Xtest, self._ydev, self._ytest = self.build_dataset(number_cat=number_cat,model=model, representation=rep,train_set=train_set, combination=combination)
        elif number_cat == "four class":
            self.pos, self.neg, self.neu, self.mixed = self.get_data(self.dataset)
            self.pos_text = self.shuffle_data(self.get_text(self.pos))
            self.neg_text = self.shuffle_data(self.get_text(self.neg))
            self.neu_text = self.shuffle_data(self.get_text(self.neu))
            self.mixed_text = self.shuffle_data(self.get_text(self.mixed))
            self.pos_span, self.neg_span, self.neu_span, self.mixed_span = self.get_span((self.pos, self.neg, self.neu, self.mixed))
            if train_set:
                self._Xtrain, self._Xdev, self._Xtest, self._ytrain, self._ydev, self._ytest = self.build_dataset(number_cat=number_cat,
                                                                                                                  model=model,
                                                                                                                  representation=rep,
                                                                                                                  train_set=train_set, combination=combination)
            else:
                if combination:
                    self._Xtest, self._ytest = self.build_dataset(number_cat=number_cat, model=model,
                                                                      representation=rep, train_set=train_set,
                                                                      combination=combination)
                else:
                    self._Xdev, self._Xtest, self._ydev, self._ytest = self.build_dataset(number_cat=number_cat,model=model, representation=rep, train_set=train_set, combination=combination)

        self.info = self.get_meta_information(self.dataset)

    def open_json(self,fname):
        """Opens a json dataset file and gets its contents"""
        with open(fname, "r") as infile:
            dataset = json.load(infile)
        return dataset

    def get_meta_information(self, dataset):
        """Given a loaded json dataset file, gets meta information from the json file and adds information about the corpus,
        including length of training, dev and test sets"""
        meta_information = dataset["meta_information"]
        if self.train_set:
            meta_information["corpora_size"] = {"total_data": len(self.pos_text) + len(self.neg_text) ,"training_set": len(self._Xtrain), "dev_set": len(self._Xdev), "test_size": len(self._Xtest)}
        else:
            if self.combination:
                meta_information["corpora_size"] = {"total_data": len(self.pos_text) + len(self.neg_text)}
            else:
                meta_information["corpora_size"] = {"total_data": len(self.pos_text) + len(self.neg_text), "dev_set": len(self._Xdev),"test_size": len(self._Xtest)}
        return meta_information

    def get_data(self, dataset):
        """Gets the data for each class, which includes doc_id, text and annotation. If binary, the classes are named positive and negative class.
        In case of concepts, positive are those examples that belong to a concept and negative those that do not.
        However, the negative class may include more than negative examples, such as neutral and mixed assessments in case of sentiment."""
        if self.number_cat == "binary":
            positive_class = dataset["positive_class"]
            negative_class = dataset["negative_class"]

            return positive_class, negative_class

        elif self.number_cat == "three class":
            positive_class = dataset["positive_class"]
            negative_class = dataset["negative_class"]
            neutral_class = dataset["neutral_class"]

            return positive_class, negative_class, neutral_class

        elif self.number_cat == "four class":
            positive_class = dataset["positive_class"]
            negative_class = dataset["negative_class"]
            neutral_class = dataset["neutral_class"]
            mixed_class = dataset["mixed_class"]

            return positive_class, negative_class, neutral_class, mixed_class

        else:
            print("Please choose a suitable number of classes for your classification task!")

    def get_text(self, dataset):
        """Gets the documents for a specific label, e.g. positive or negative data obtained with get_data()"""
        data = []
        format_data = []
        for elem in dataset:
            data.append(elem["text"])
        for doc in data:
            # Commented code applies only if representation=words in the older form, where words = sentence.split
            #doc = re.sub('(?<! )(?=[.;:,!?()>])|(?<=[.,;:!?()>])(?! )', r' ', doc)
            doc = doc.strip('"').strip("'")
            format_data.append(doc)
        return format_data

    def get_docid(self, dataset):
        """Gets the docID for the specific label"""
        doc_ids = []
        for elem in dataset:
            doc_ids.append(elem["doc_id"])

        return doc_ids

    def get_span(self,texts):
        """Gets the spans for each document."""
        #TODO: Is this still needed?
        pos_spans = []
        neg_spans = []
        neu_spans = []
        mixed_spans = []
        if self.number_cat == "binary":
            for elem in texts[0]:
                if "span" in elem.keys():
                    pos_spans.append(elem["span"])
                else:
                    pos_spans.append([])
            for elem in texts[1]:
                if "span" in elem.keys():
                    neg_spans.append(elem["span"])
                else:
                    neg_spans.append([])
            return pos_spans, neg_spans

        elif self.number_cat == "three class":
            for elem in texts[0]:
                if "span" in elem.keys():
                    pos_spans.append(elem["span"])
                else:
                    pos_spans.append([])
            for elem in texts[1]:
                if "span" in elem.keys():
                    neg_spans.append(elem["span"])
                else:
                    neg_spans.append([])
            for elem in texts[2]:
                if "span" in elem.keys():
                    neu_spans.append(elem["span"])
                else:
                    neu_spans.append([])

            return pos_spans, neg_spans, neu_spans

        elif self.number_cat == "four class":
            for elem in texts[0]:
                if "span" in elem.keys():
                    pos_spans.append(elem["span"])
                else:
                    pos_spans.append([])
            for elem in texts[1]:
                if "span" in elem.keys():
                    neg_spans.append(elem["span"])
                else:
                    neg_spans.append([])
            for elem in texts[2]:
                if "span" in elem.keys():
                    neu_spans.append(elem["span"])
                else:
                    neu_spans.append([])
            for elem in texts[3]:
                if "span" in elem.keys():
                    mixed_spans.append(elem["span"])
                else:
                    mixed_spans.append([])

            return pos_spans, neg_spans, neu_spans, mixed_spans

    def shuffle_data(self, text):
        """Shuffles the text. Note that a seed is set, so that the same result is always obtained."""
        random.seed(42)
        random.shuffle(text)
        return text

    def get_tokens_labels(self, cat_text, number_cat, model, representation=ave_vecs):
        """ Tokenizes the sentences and provides them with a label"""
        # Encoding might have to be changed for non-western languages. Previously encoding="latin"
        if number_cat == "binary":
            return getMyData(cat_text[0], 0, model, representation, encoding='utf8'), getMyData(cat_text[1], 1, model,
                                                                                                representation,
                                                                                                encoding='utf8')
        elif number_cat == "three class":
            return getMyData(cat_text[0], 0, model, representation, encoding='utf8'), getMyData(cat_text[1], 1, model,
                                                                                                representation,
                                                                                                encoding='utf8'), getMyData(
                cat_text[2], 2, model, representation, encoding='utf8')
        elif number_cat == "four class":
            return getMyData(cat_text[0], 0, model, representation, encoding='utf8'), getMyData(cat_text[1], 1, model,
                                                                                                representation,
                                                                                                encoding='utf8'), getMyData(
                cat_text[2], 2, model, representation, encoding='utf8'), getMyData(cat_text[3], 3, model,
                                                                                   representation, encoding='utf-8')

        else:
            print(
                "Assignment of labels to texts not possible. Please make sure that the number of classes if specified correctly!")

    def train_dev_test_split(self,text):
        """ Takes a list of texts and splits the list into train, dev and testsets.
        The default proportions are 80% for the training set, 10% for the dev set and 10% for the test set"""

        split_1 = int(0.8 * len(text))
        split_2 = int(0.9 * len(text))

        train_set = text[:split_1]
        dev_set = text[split_1:split_2]
        test_set = text[split_2:]

        return train_set, dev_set, test_set

    def dev_test_split(self,text):
        """ Takes a list of texts and splits the list into dev and testsets.
        The default proportions are 50% for the dev set and 50% for the test set"""
        split_1 = int(0.5 * len(text))

        dev_set = text[:split_1]
        test_set = text[split_1:]

        return dev_set, test_set

    def to_array(self, integer, num_labels):
        """quick trick to convert an integer to a one hot vector that
            corresponds to the y labels"""
        integer = integer - 1
        return np.array(np.eye(num_labels)[integer])

    def build_dataset(self, number_cat, model, representation=ave_vecs, train_set=True, combination=False):
        """Makes the datasets ready for use in BLSE. If a train-set is available, thee splitting is done according to
        train_dev_test_split() and if not according to dev_test_split().
        Sets up the vocabulary, text and labels for each dataset """

        if number_cat == "binary":
            labeled_pos_data,labeled_neg_data = self.get_tokens_labels((self.pos_text,self.neg_text),number_cat=number_cat,model=model,representation=representation)
            if train_set:
                train_pos, dev_pos, test_pos = self.train_dev_test_split(labeled_pos_data)
                train_neg, dev_neg, test_neg = self.train_dev_test_split(labeled_neg_data)

                traindata = train_pos + train_neg
                devdata = dev_pos + dev_neg
                testdata = test_pos + test_neg
                total_dataset = traindata + devdata + testdata

                #shuffle the data again
                traindata = random.sample(traindata, len(traindata))
                devdata = random.sample(devdata, len(devdata))
                testdata = random.sample(testdata, len(testdata))

                # Set up vocab now
                self.vocab = set()

                #Training data
                Xtrain = [data for data, y in traindata]
                if self.lowercase:
                    Xtrain = [[w.lower() for w in sent] for sent in Xtrain]
                if self.one_hot is True:
                    ytrain = [self.to_array(y, 2) for data, y in traindata]
                else:
                    ytrain = [y for data, y in traindata]
                self.vocab.update(set([w for i in Xtrain for w in i]))

                # Dev data
                Xdev = [data for data, y in devdata]
                if self.lowercase:
                    Xdev = [[w.lower() for w in sent] for sent in Xdev]
                if self.one_hot is True:
                    ydev = [self.to_array(y, 2) for data, y in devdata]
                else:
                    ydev = [y for data, y in devdata]
                self.vocab.update(set([w for i in Xdev for w in i]))

                # Test data
                Xtest = [data for data, y in testdata]
                if self.lowercase:
                    Xtest = [[w.lower() for w in sent] for sent in Xtest]
                if self.one_hot is True:
                    ytest = [self.to_array(y, 2) for data, y in testdata]
                else:
                    ytest = [y for data, y in testdata]
                self.vocab.update(set([w for i in Xtest for w in i]))

                if self.rep is not words:
                    Xtrain = np.array(Xtrain)
                    Xdev = np.array(Xdev)
                    Xtest = np.array(Xtest)
                ytrain = np.array(ytrain)
                ydev = np.array(ydev)
                ytest = np.array(ytest)

                return Xtrain, Xdev, Xtest, ytrain, ydev, ytest

            else:
                if combination:
                    test_pos = labeled_pos_data
                    test_neg = labeled_neg_data

                    testdata = test_pos + test_neg
                    total_dataset = testdata

                    # Set up vocab now
                    self.vocab = set()

                    # Test data
                    Xtest = [data for data, y in testdata]
                    if self.lowercase:
                        Xtest = [[w.lower() for w in sent] for sent in Xtest]
                    if self.one_hot is True:
                        ytest = [self.to_array(y, 2) for data, y in testdata]
                    else:
                        ytest = [y for data, y in testdata]
                    self.vocab.update(set([w for i in Xtest for w in i]))

                    if self.rep is not words:
                        Xtest = np.array(Xtest)
                    ytest = np.array(ytest)

                    test_pos_id = self.pos_ids
                    test_neg_id = self.neg_ids
                    docid = test_pos_id + test_neg_id
                    ##docid = random.sample(docid, len(docid))

                    return Xtest, ytest, docid

                else:
                    dev_pos, test_pos = self.dev_test_split(labeled_pos_data)
                    dev_neg, test_neg = self.dev_test_split(labeled_neg_data)

                    #split docid pos and neg
                    dev_pos_id, test_pos_id = self.dev_test_split(self.pos_ids)
                    dev_neg_id, test_neg_id = self.dev_test_split(self.neg_ids)

                    devdata = dev_pos + dev_neg
                    testdata = test_pos + test_neg
                    total_dataset = devdata + testdata

                    devid = dev_pos_id + dev_neg_id
                    testid = test_pos_id + test_neg_id

                    #shuffling of the dev and test data and id
                    devdata = random.sample(devdata, len(devdata))
                    testdata = random.sample(testdata, len(testdata))

                    devid = random.sample(devid, len(devid))
                    testid = random.sample(testid, len(testid))

                    # Set up vocab now
                    self.vocab = set()


                    # Dev data
                    Xdev = [data for data, y in devdata]
                    if self.lowercase:
                        Xdev = [[w.lower() for w in sent] for sent in Xdev]
                    if self.one_hot is True:
                        ydev = [self.to_array(y, 2) for data, y in devdata]
                    else:
                        ydev = [y for data, y in devdata]
                    self.vocab.update(set([w for i in Xdev for w in i]))

                    # Test data
                    Xtest = [data for data, y in testdata]
                    if self.lowercase:
                        Xtest = [[w.lower() for w in sent] for sent in Xtest]
                    if self.one_hot is True:
                        ytest = [self.to_array(y, 2) for data, y in testdata]
                    else:
                        ytest = [y for data, y in testdata]
                    self.vocab.update(set([w for i in Xtest for w in i]))


                    if self.rep is not words:
                        Xdev = np.array(Xdev)
                        Xtest = np.array(Xtest)
                    ydev = np.array(ydev)
                    ytest = np.array(ytest)

                    return Xdev, Xtest, ydev, ytest, devid, testid

        elif number_cat == "three class":
            labeled_pos_data, labeled_neg_data, labeled_neu_data = self.get_tokens_labels((self.pos_text, self.neg_text,self.neu_text),number_cat=number_cat, model=model,
                                                            representation=representation)
            if train_set:
                train_pos, dev_pos, test_pos = self.train_dev_test_split(labeled_pos_data)
                train_neg, dev_neg, test_neg = self.train_dev_test_split(labeled_neg_data)
                train_neu, dev_neu, test_neu = self.train_dev_test_split(labeled_neu_data)

                traindata = train_pos + train_neg + train_neu
                devdata = dev_pos + dev_neg + dev_neu
                testdata = test_pos + test_neg + test_neu
                total_dataset = traindata + devdata + testdata

                # Set up vocab now
                self.vocab = set()

                # Training data
                Xtrain = [data for data, y in traindata]
                if self.lowercase:
                    Xtrain = [[w.lower() for w in sent] for sent in Xtrain]
                if self.one_hot is True:
                    ytrain = [self.to_array(y, 3) for data, y in traindata]
                else:
                    ytrain = [y for data, y in traindata]
                self.vocab.update(set([w for i in Xtrain for w in i]))

                # Dev data
                Xdev = [data for data, y in devdata]
                if self.lowercase:
                    Xdev = [[w.lower() for w in sent] for sent in Xdev]
                if self.one_hot is True:
                    ydev = [self.to_array(y, 3) for data, y in devdata]
                else:
                    ydev = [y for data, y in devdata]
                self.vocab.update(set([w for i in Xdev for w in i]))

                # Test data
                Xtest = [data for data, y in testdata]
                if self.lowercase:
                    Xtest = [[w.lower() for w in sent] for sent in Xtest]
                if self.one_hot is True:
                    ytest = [self.to_array(y, 3) for data, y in testdata]
                else:
                    ytest = [y for data, y in testdata]
                self.vocab.update(set([w for i in Xtest for w in i]))

                return Xtrain, Xdev, Xtest, ytrain, ydev, ytest

            else:

                dev_pos, test_pos = self.dev_test_split(labeled_pos_data)
                dev_neg, test_neg = self.dev_test_split(labeled_neg_data)
                dev_neu, test_neu = self.dev_test_split(labeled_neu_data)

                devdata = dev_pos + dev_neg + dev_neu
                testdata = test_pos + test_neg + test_neu
                total_dataset = devdata + testdata

                # Set up vocab now
                self.vocab = set()

                # Dev data
                Xdev = [data for data, y in devdata]
                if self.lowercase:
                    Xdev = [[w.lower() for w in sent] for sent in Xdev]
                if self.one_hot is True:
                    ydev = [self.to_array(y, 3) for data, y in devdata]
                else:
                    ydev = [y for data, y in devdata]
                self.vocab.update(set([w for i in Xdev for w in i]))

                # Test data
                Xtest = [data for data, y in testdata]
                if self.lowercase:
                    Xtest = [[w.lower() for w in sent] for sent in Xtest]
                if self.one_hot is True:
                    ytest = [self.to_array(y, 3) for data, y in testdata]
                else:
                    ytest = [y for data, y in testdata]
                self.vocab.update(set([w for i in Xtest for w in i]))

                return Xdev, Xtest, ydev, ytest

        elif number_cat == "four class":
            labeled_pos_data, labeled_neg_data, labeled_neu_data, labeled_mixed_data = self.get_tokens_labels(
                (self.pos_text, self.neg_text, self.neu_text, self.mixed_text), number_cat=number_cat, model=model,
                representation=representation)
            if train_set:
                train_pos, dev_pos, test_pos = self.train_dev_test_split(labeled_pos_data)
                train_neg, dev_neg, test_neg = self.train_dev_test_split(labeled_neg_data)
                train_neu, dev_neu, test_neu = self.train_dev_test_split(labeled_neu_data)
                train_mixed, dev_mixed, test_mixed = self.train_dev_test_split(labeled_mixed_data)

                traindata = train_pos + train_neg + train_neu + train_mixed
                devdata = dev_pos + dev_neg + dev_neu + dev_mixed
                testdata = test_pos + test_neg + test_neu + test_mixed
                total_dataset = traindata + devdata + testdata

                # Set up vocab now
                self.vocab = set()

                # Training data
                Xtrain = [data for data, y in traindata]
                if self.lowercase:
                    Xtrain = [[w.lower() for w in sent] for sent in Xtrain]
                if self.one_hot is True:
                    ytrain = [self.to_array(y, 4) for data, y in traindata]
                else:
                    ytrain = [y for data, y in traindata]
                self.vocab.update(set([w for i in Xtrain for w in i]))

                # Dev data
                Xdev = [data for data, y in devdata]
                if self.lowercase:
                    Xdev = [[w.lower() for w in sent] for sent in Xdev]
                if self.one_hot is True:
                    ydev = [self.to_array(y, 4) for data, y in devdata]
                else:
                    ydev = [y for data, y in devdata]
                self.vocab.update(set([w for i in Xdev for w in i]))

                # Test data
                Xtest = [data for data, y in testdata]
                if self.lowercase:
                    Xtest = [[w.lower() for w in sent] for sent in Xtest]
                if self.one_hot is True:
                    ytest = [self.to_array(y, 4) for data, y in testdata]
                else:
                    ytest = [y for data, y in testdata]
                self.vocab.update(set([w for i in Xtest for w in i]))

                if self.rep is not words:
                    Xtrain = np.array(Xtrain)
                    Xdev = np.array(Xdev)
                    Xtest = np.array(Xtest)
                ytrain = np.array(ytrain)
                ydev = np.array(ydev)
                ytest = np.array(ytest)

                return Xtrain, Xdev, Xtest, ytrain, ydev, ytest

            else:
                dev_pos, test_pos = self.dev_test_split(labeled_pos_data)
                dev_neg, test_neg = self.dev_test_split(labeled_neg_data)
                dev_neu, test_neu = self.dev_test_split(labeled_neu_data)
                dev_mixed, test_mixed = self.dev_test_split(labeled_mixed_data)

                devdata = dev_pos + dev_neg + dev_neu + dev_mixed
                testdata = test_pos + test_neg + test_neu + test_mixed
                total_dataset = devdata + testdata

                # Set up vocab now
                self.vocab = set()

                # Dev data
                Xdev = [data for data, y in devdata]
                if self.lowercase:
                    Xdev = [[w.lower() for w in sent] for sent in Xdev]
                if self.one_hot is True:
                    ydev = [self.to_array(y, 4) for data, y in devdata]
                else:
                    ydev = [y for data, y in devdata]
                self.vocab.update(set([w for i in Xdev for w in i]))

                # Test data
                Xtest = [data for data, y in testdata]
                if self.lowercase:
                    Xtest = [[w.lower() for w in sent] for sent in Xtest]
                if self.one_hot is True:
                    ytest = [self.to_array(y, 4) for data, y in testdata]
                else:
                    ytest = [y for data, y in testdata]
                self.vocab.update(set([w for i in Xtest for w in i]))

                if self.rep is not words:
                    Xdev = np.array(Xdev)
                    Xtest = np.array(Xtest)
                ydev = np.array(ydev)
                ytest = np.array(ytest)

                return Xdev, Xtest, ydev, ytest


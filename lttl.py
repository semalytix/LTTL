import argparse
import datetime
import re
import sys
import yaml
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from Utils.Datasets import *
from Utils.WordVecs import *
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

run_time = datetime.datetime.now()

class LTTL(nn.Module):
    """
    Language and task informed transfer learning framework based on Bilingual Sentiment Embeddings

    Parameters:

        src_vecs: WordVecs instance with the embeddings from the source language
        trg_vecs: WordVecs instance with the embeddings from the target language
        pdataset: Projection_Dataset from source to target language (lexicon)
        cdataset: Source dataset
        projection_loss: the distance metric to use for the projection loss
                         can be either mse (default) or cosine
        output_dim: the number of class labels to predict (default: 4)
    """

    def __init__(self, src_vecs, trg_vecs, trg_dataset=None,
                 pdataset=None, cdataset=None,
                 projection_loss=None,
                 output_dim=4,
                 summary_writter_params=None
                 ):
        super(LTTL, self).__init__()

        if cdataset:

            # Embedding matrices
            self.semb = nn.Embedding(src_vecs.vocab_length, src_vecs.vector_size)
            self.semb.weight.data.copy_(torch.from_numpy(src_vecs._matrix))  # comment for randomized vectors
            self.sw2idx = src_vecs._w2idx
            self.sidx2w = src_vecs._idx2w
            self.temb = nn.Embedding(trg_vecs.vocab_length, trg_vecs.vector_size)
            self.temb.weight.data.copy_(torch.from_numpy(trg_vecs._matrix))  # comment for randomized vectors
            self.tw2idx = trg_vecs._w2idx
            self.tidx2w = trg_vecs._idx2w

            # Adding embeddings to the tensorboard
            self.writer = SummaryWriter(log_dir=summary_writter_params)
            logger.info("Comparing the size of the vocabulary source and target embeddings")
            logger.info(f"This is the source embeddings vocabulary length: {src_vecs.vocab_length}")
            logger.info(f"This is the target embeddings vocabulary length: {trg_vecs.vocab_length}")

            # Chooses output size if the src and trg vectors have different dimension size
            logger.info("Comparing the number of dimensions for source and target embeddings")
            if src_vecs.vector_size == trg_vecs.vector_size:
                output_vector_size = src_vecs.vector_size
                logger.info(f"Vectors have the same size : {src_vecs.vector_size}")
            else:
                logger.info(f"Vectors don't have the same size.")
                logger.info(f"source vector : {src_vecs.vector_size}, target vector : {trg_vecs.vector_size}")
                vector_sizes = [src_vecs.vector_size, trg_vecs.vector_size]
                output_vector_size = min(vector_sizes)
                logger.info(f"Final vector size: {output_vector_size}")

            vector_info = json.dumps(
                {"source vector size": src_vecs.vector_size, "target vector size": trg_vecs.vector_size,
                 "output vector size": output_vector_size})
            self.vector_info = vector_info

            # Projection vectors (the output vector size correspondes to the smallest embedding space if the embeddings have different dimensionalities)
            # output vector size is the size of the source embedding

            self.m = nn.Linear(src_vecs.vector_size,
                               output_vector_size,
                               bias=False)

            self.mp = nn.Linear(trg_vecs.vector_size,
                                output_vector_size,
                                bias=False)

            # Classifier
            self.clf = nn.Linear(src_vecs.vector_size, output_dim)

            # Loss Functions
            self.criterion = nn.CrossEntropyLoss()  # Loss function for the classification
            if projection_loss == 'mse':
                self.proj_criterion = mse_loss  # Loss function for the linear projection (M - M')
            elif projection_loss == 'cosine':
                self.proj_criterion = cosine_loss

            # Optimizer
            self.optim = torch.optim.Adam(self.parameters())
            # self.optim = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

            # Datasets
            self.pdataset = pdataset
            self.cdataset = cdataset
            self.trg_dataset = trg_dataset


            # Trg Data
            if self.trg_dataset != None:
                self.trg_data = True
            else:
                self.trg_data = False

            # History
            self.history = {'loss': [], 'dev_cosine': [], 'dev_f1': [], 'cross_f1': [],
                            'syn_cos': [], 'ant_cos': [], 'cross_syn': [], 'cross_ant': [],
                            'dev_prec': [], 'dev_rec': [], 'dev_acc': [], 'cross_prec': [], 'cross_rec': [],
                            'cross_acc': []}

            # Do not update original embedding spaces
            self.semb.weight.requires_grad = False
            self.temb.weight.requires_grad = False

        else:

            # Embedding matrices
            self.semb = nn.Embedding(src_vecs.vocab_length, src_vecs.vector_size)
            self.semb.weight.data.copy_(torch.from_numpy(src_vecs._matrix))  # comment for randomized vectors
            self.sw2idx = src_vecs._w2idx
            self.sidx2w = src_vecs._idx2w
            self.temb = nn.Embedding(trg_vecs.vocab_length, trg_vecs.vector_size)
            self.temb.weight.data.copy_(torch.from_numpy(trg_vecs._matrix))  # comment for randomized vectors
            self.tw2idx = trg_vecs._w2idx
            self.tidx2w = trg_vecs._idx2w

            # Chooses output size if the src and trg vectors have different dimension size
            logger.info("Comparing the number of dimensions for source and target embeddings")
            if src_vecs.vector_size == trg_vecs.vector_size:
                output_vector_size = src_vecs.vector_size
                logger.info(f"Vectors have the same size : {src_vecs.vector_size}")
            else:
                logger.info(f"Vectors don't have the same size.")
                logger.info(f"source vector : {src_vecs.vector_size}, target vector : {trg_vecs.vector_size}")
                vector_sizes = [src_vecs.vector_size, trg_vecs.vector_size]
                output_vector_size = min(vector_sizes)
                logger.info(f"Final vector size: {output_vector_size}")

            # Projection vectors (the output vector size correspondes to the smallest embedding space if the embeddings have different dimensionalities)
            self.m = nn.Linear(src_vecs.vector_size,
                               output_vector_size,
                               bias=False)

            self.mp = nn.Linear(trg_vecs.vector_size,
                                output_vector_size,
                                bias=False)

            # Classifier
            self.clf = nn.Linear(src_vecs.vector_size, output_dim)

            self.trg_dataset = trg_dataset

    def dump_weights(self, outfile):
        # Dump the weights to outfile
        w1 = self.m.weight.data.numpy()
        w2 = self.mp.weight.data.numpy()
        w3 = self.clf.weight.data.numpy()
        b = self.clf.bias.data.numpy()
        np.savez(outfile, w1, w2, w3, b)

    def load_weights(self, weight_file):
        # Load weights from weight_file
        f = np.load(weight_file)
        w1 = self.m.weight.data.copy_(torch.from_numpy(f['arr_0']))
        w2 = self.mp.weight.data.copy_(torch.from_numpy(f['arr_1']))
        w3 = self.clf.weight.data.copy_(torch.from_numpy(f['arr_2']))
        b = self.clf.bias.data.copy_(torch.from_numpy(f['arr_3']))

    def project(self, X, Y):
        """
        Project X and Y into shared space.
        X is a list of source words from the projection lexicon,
        and Y is the list of single word translations.
        """
        x_lookup = torch.LongTensor(np.array([self.sw2idx[w] for w in X]))
        y_lookup = torch.LongTensor(np.array([self.tw2idx[w] for w in Y]))
        x_embedd = self.semb(Variable(x_lookup))
        y_embedd = self.temb(Variable(y_lookup))
        x_proj = self.m(x_embedd)
        y_proj = self.mp(y_embedd)
        return x_proj, y_proj

    def look_up_meta_info_for_results(self, X, Y):
        """
        Tracks the coverage of the translation lexicon.
        For each pair, it checks if there are corresponding embeddings in the source and target language
        """
        x_lookup = torch.LongTensor(np.array([self.sw2idx[w] for w in X]))
        y_lookup = torch.LongTensor(np.array([self.tw2idx[w] for w in Y]))
        x_embedd = self.semb(Variable(x_lookup))
        y_embedd = self.temb(Variable(y_lookup))
        x_proj = self.m(x_embedd)
        y_proj = self.mp(y_embedd)
        lexicon_src_matches = list(x_proj.shape)[0]
        lexicon_trg_matches = list(y_proj.shape)[0]
        look_up_infos = json.dumps({"coverage src embeddings/lexicon": lexicon_src_matches,
                                    "coverage trg embeddings/lexicon": lexicon_trg_matches})
        return look_up_infos

    def projection_loss(self, x, y):
        """
        Find the loss between the two projected sets of translations.
        The loss is the proj_criterion.
        """

        x_proj, y_proj = self.project(x, y)

        # distance-based loss (cosine, mse)
        loss = self.proj_criterion(x_proj, y_proj)

        return loss

    def idx_vecs(self, sentence, model):
        """
        Converts a tokenized sentence to a vector
        of word indices based on the model.
        """
        sent = []
        for w in sentence:
            try:
                sent.append(model[w])
            except KeyError:
                sent.append(0)
        return torch.LongTensor(sent)

    def lookup(self, X, model):
        """
        Converts a batch of tokenized sentences
        to a matrix of word indices from model.
        """
        return [self.idx_vecs(s, model) for s in X]

    def ave_vecs(self, X, src=True):
        """
        Converts a batch of tokenized sentences into
        a matrix of averaged word embeddings. If src
        is True, it uses the sw2idx and semb to create
        the averaged representation. Otherwise, it uses
        the tw2idx and temb.
        """
        vecs = []
        if src:
            idxs = self.lookup(X, self.sw2idx)
            for i in idxs:
                vecs.append(self.semb(Variable(i)).mean(0))
        else:
            idxs = self.lookup(X, self.tw2idx)
            for i in idxs:
                vecs.append(self.temb(Variable(i)).mean(0))
        return torch.stack(vecs)

    def predict(self, X, src=True):
        """
        Projects the averaged embeddings from X
        to the joint space and then uses either
        m (if src==True) or mp (if src==False)
        to predict the sentiment of X.
        """

        X = self.ave_vecs(X, src)
        if src:
            x_proj = self.m(X)
        else:
            x_proj = self.mp(X)
        out = F.softmax(self.clf(x_proj))
        return out

    def classification_loss(self, x, y, src=True):
        pred = self.predict(x, src=src)
        y = Variable(torch.from_numpy(y))
        loss = self.criterion(pred, y)
        return loss

    def full_loss(self, proj_x, proj_y, class_x, class_y,
                  alpha=.5):
        """
        Calculates the combined projection and classification loss.
        Alpha controls the amount of weight given to each loss term.
        """

        proj_loss = self.projection_loss(proj_x, proj_y)
        #logger.info("projection loss :"+proj_loss)
        class_loss = self.classification_loss(class_x, class_y, src=True)
        #logger.info("classification loss :"+class_loss)
        return alpha * proj_loss + (1 - alpha) * class_loss

    def shuffle_dataset(self, X, y):
        """
        Shuffles X and y while preserving their correspondent order
        """
        data_label = list(zip(X, y))
        random.seed(42)
        random.shuffle(data_label)
        out_X, out_y = zip(*data_label)
        return list(out_X), np.array(list(out_y))

    def fit(self, proj_X, proj_Y,
            class_X, class_Y,
            weight_dir='models',
            batch_size=40,
            epochs=100,
            alpha=0.5):

        """
        Trains the model on the projection data (and
        source language sentiment data (class_X, class_Y).
        """

        num_batches = int(len(class_X) / batch_size)
        best_cross_f1 = 0
        num_epochs = 0

        # Shuffle classification dataset for every epoch
        class_X, class_Y = self.shuffle_dataset(class_X, class_Y)

        for i in range(epochs):
            idx = 0
            num_epochs += 1

            for j in range(num_batches):
                cx = class_X[idx:idx + batch_size]
                cy = class_Y[idx:idx + batch_size]
                idx += batch_size
                self.optim.zero_grad()
                loss = self.full_loss(proj_X, proj_Y, cx, cy, alpha)
                loss.backward()
                self.optim.step()
                #logger.info(loss.item())

                # check cosine distance between dev translation pairs
                xdev = self.pdataset._Xdev
                ydev = self.pdataset._ydev
                xp, yp = self.project(xdev, ydev)
                score = cos(xp, yp)

                # check source dev f1
                xdev = self.cdataset._Xdev
                ydev = self.cdataset._ydev
                xp = self.predict(xdev).data.numpy().argmax(1)

                # macro f1
                dev_f1 = macro_f1(ydev, xp)
                #logger.info("Source Dataset F1 " + str(dev_f1))

                # precision
                dev_prec = precision_score(ydev, xp, average="macro")
                #logger.info("Source Dataset Precision " + str(dev_prec))

                # recall
                dev_rec = recall_score(ydev, xp, average="macro")
                #logger.info("Source Dataset Recall " + str(dev_rec))

                # accuracy
                dev_acc = accuracy_score(ydev, xp)
                #logger.info("Source Dataset Accuracy " + str(dev_acc))

                # confusion matrix
                src_cm = confusion_matrix(ydev, xp)
                tn, fp, fn, tp = src_cm.ravel()
                #logger.info(src_cm)
                sum_examples = (tn + fp + fn + tp)
                #logger.info(f"True positives: {tp}, True negatives: {tn}, False positives: {fp}, False negatives: {fn}")
                #logger.info(f"There were {sum_examples} documents")

            for j in range(num_batches):
                dev_f1 = macro_f1(ydev, xp)
                #logger.info(f"source dev_f1 {dev_f1} (dev translation pairs)")

            if self.trg_data:
                # check target dev f1
                crossx = self.trg_dataset._Xdev
                crossy = self.trg_dataset._ydev
                xp = self.predict(crossx, src=False).data.numpy().argmax(1)

                # macro f1
                cross_f1 = macro_f1(crossy, xp)
                #logger.info("Target Dataset F1 " + str(cross_f1))

                # precision
                cross_prec = precision_score(crossy, xp, average="macro")
                #logger.info("Target Dataset Precision " + str(cross_prec))

                # recall
                cross_rec = recall_score(crossy, xp, average="macro")
                #logger.info("Target Dataset Recall " + str(cross_rec))

                # accuracy
                cross_acc = accuracy_score(crossy, xp)
                #logger.info("Target Dataset Accuracy" + str(cross_acc))

                trg_cm = confusion_matrix(crossy, xp)
                tn, fp, fn, tp = trg_cm.ravel()
                #logger.info(trg_cm)
                sum_examples = (tn + fp + fn + tp)
                #logger.info(f"True positives: {tp}, True negatives: {tn}, False positives: {fp}, False negatives: {fn}")
                #logger.info(f"There were {sum_examples} documents")
                #logger.info(f"Epoch: {num_epochs}")

                run_time_now = datetime.datetime.now()
                if cross_f1 > best_cross_f1:
                    best_cross_f1 = cross_f1
                    weight_file = os.path.join(weight_dir, run_time_now.strftime(
                        "%d-%m-%Y%H:%M:%S") + "-" + '{0}epochs-{1}batchsize-{2}alpha-{3:.3f}crossf1'.format(num_epochs,
                                                                                                            batch_size,
                                                                                                            alpha,
                                                                                                            best_cross_f1))
                    self.dump_weights(weight_file)
                elif num_epochs == 200:
                    weight_file = os.path.join(weight_dir, run_time_now.strftime("%d-%m-%Y%H:%M:%S") + "-" +
                                               '{0}epochs-{1}batchsize-{2}alpha-{3:.3f}crossf1'.format(num_epochs,
                                                                                                       batch_size,
                                                                                                       alpha,
                                                                                                       cross_f1))
                    self.dump_weights(weight_file)

                # saves the metrics into a history
                sys.stdout.flush()
                self.history['loss'].append(loss.data.item())
                self.history['dev_cosine'].append(score.data.item())
                self.history['dev_f1'].append(dev_f1)
                self.history['cross_f1'].append(cross_f1)

                self.history['dev_prec'].append(dev_prec)
                self.history['dev_rec'].append(dev_rec)
                self.history['dev_acc'].append(dev_acc)
                self.history['cross_prec'].append(cross_prec)
                self.history['cross_rec'].append(cross_rec)
                self.history['cross_acc'].append(cross_acc)

                # logs the evaluation metrics to the tensorboard for each run
                self.writer.add_scalar("loss", loss.data.item(), num_epochs)
                self.writer.add_scalar("dev_cosine", score.data.item(), num_epochs)
                self.writer.add_scalar("source f1 (dev)", dev_f1, num_epochs)
                self.writer.add_scalar("source precision (dev)", dev_prec, num_epochs)
                self.writer.add_scalar("source recall (dev)", dev_rec, num_epochs)
                self.writer.add_scalar("source accuracy (dev)", dev_acc, num_epochs)
                self.writer.add_scalar("target f1 (test)", cross_f1, num_epochs)
                self.writer.add_scalar("target precision (test)", cross_prec, num_epochs)
                self.writer.add_scalar("target recall (test)", cross_rec, num_epochs)
                self.writer.add_scalar("target accuracy (test)", cross_acc, num_epochs)
                self.writer.add_scalars("evaluation training",
                                        {"source f1 (dev)": dev_f1,
                                         "source precision (dev)": dev_prec,
                                         "source recall (dev)": dev_rec,
                                         "source accuracy (dev)": dev_acc,
                                         }, num_epochs)
                self.writer.add_scalars("evaluation test",
                                        {"target f1 (test)": cross_f1,
                                         "target precision (test)": cross_prec,
                                         "target recall (test)": cross_rec,
                                         "target accuracy (test)": cross_acc
                                         }, num_epochs)


    def confusion_matrix(self, X, Y, src=True):
        """
        Prints a confusion matrix for the model
        """

        pred = self.predict(X, src=src).data.numpy().argmax(1)
        cm = confusion_matrix(Y, pred, sorted(set(Y)))
        print("Confusion Matrix" + "\n" + str(cm))
        tn, fp, fn, tp = cm.ravel()
        sum_examples = (tn + fp + fn + tp)
        print(f"True positives: {tp}, True negatives: {tn}, False positives: {fp}, False negatives: {fn}")
        print(f"There were {sum_examples} documents")

    def evaluate(self, X, Y, src=True, outfile=None):
        """
        Prints the accuracy, macro precision, macro recall,
        and macro F1 of the model on X. If outfile != None,
        the predictions are written to outfile.
        """

        pred = self.predict(X, src=src).data.numpy().argmax(1)
        acc = accuracy_score(Y, pred)  # accuracy score (real label, prediction label)
        prec = per_class_prec(Y, pred).mean()  # precision of (real label, prediction label)
        rec = per_class_rec(Y, pred).mean()  # recall of (real label, prediction label)
        f1 = macro_f1(Y, pred)  # macro f1 score of (real label, prediction label)
        report = classification_report(Y, pred, output_dict=True)
        if outfile:
            with open(outfile, 'w') as out:
                # outputs the pred file with pred only
                for i in pred:
                    out.write('{0}\n'.format(i))

        else:
            print(' Test Set:')
            print(
                'acc:  {0:.3f}\nmacro prec: {1:.3f}\nmacro rec: {2:.3f}\nmacro f1: {3:.3f}'.format(acc, prec, rec, f1))
        test_set_results = {}
        if src:
            test_set_results["Source F1"] = "{0:.3f}".format(f1)
            test_set_results["Source Accuracy"] = "{0:.3f}".format(acc)
            test_set_results["Source Precision"] = "{0:.3f}".format(prec)
            test_set_results["Source Recall"] = "{0:.3f}".format(rec)

        else:
            test_set_results["Target F1"] = "{0:.3f}".format(f1)
            test_set_results["Target Accuracy"] = "{0:.3f}".format(acc)
            test_set_results["Target Precision"] = "{0:.3f}".format(prec)
            test_set_results["Target Recall"] = "{0:.3f}".format(rec)
            test_set_results["Report"] = report
        test_results = json.dumps(test_set_results, indent=2)

        return test_results

    def evaluation_preds(self, X, test_ids, outfile, src=True):
        """
        the predictions are written to outfile.
        """

        logger.info("Precictions are being written to: "+outfile)

        pred = self.predict(X, src=src).data.numpy().argmax(1)

        if len(test_ids) == len(pred):
            with open(outfile, 'w') as out:
                for i in range(len(pred)):
                    t = test_ids[i]
                    p = pred[i]
                    out.write('{0}\t{1}\n'.format(t, p))
        else:
            logger.warning("Problems with nr of predictions!")

    # Takes results from history variable
    def evaluation_results(self):
        h = self.history
        #print(h.keys())
        source_f1 = h['dev_f1'][-1]  # Returns the last f1 from the development set in the source language
        target_f1 = h['cross_f1'][-1]  # Returns the last f1 from the dev set in the target language
        source_prec = h['dev_prec'][-1]
        source_rec = h['dev_rec'][-1]
        source_acc = h['dev_acc'][-1]
        target_prec = h['cross_prec'][-1]
        target_rec = h['cross_rec'][-1]
        target_acc = h['cross_acc'][-1]
        print("Source F1 (dev): {0:.3f}".format(source_f1))
        print("Target F1 (test): {0:.3f}".format(target_f1))
        print("Source Precision (dev): {0:.3f}".format(source_prec))
        print("Source Recall (dev): {0:.3f}".format(source_rec))
        print("Source Accuracy (dev): {0:.3f}".format(source_acc))
        print("Target Precision (test): {0:.3f}".format(target_prec))
        print("Target Recall (test): {0:.3f}".format(target_rec))
        print("Target Accuracy (test): {0:.3f}".format(target_acc))

        # Saves the results into a json dump to generate results files
        results = {}
        results["Source F1"] = "{0:.3f}".format(source_f1)
        results["Source Accuracy"] = "{0:.3f}".format(source_acc)
        results["Source Precision"] = "{0:.3f}".format(source_prec)
        results["Source Recall"] = "{0:.3f}".format(source_rec)
        results["Target F1"] = "{0:.3f}".format(target_f1)
        results["Target Accuracy"] = "{0:.3f}".format(target_acc)
        results["Target Precision"] = "{0:.3f}".format(target_prec)
        results["Target Recall"] = "{0:.3f}".format(target_rec)
        evaluation_results = json.dumps(results, indent=2)
        print("Target Language Test Set: ")
        print(
            'acc:  {0:.3f}\nmacro prec: {1:.3f}\nmacro rec: {2:.3f}\nmacro f1: {3:.3f}'.format(target_acc, target_prec,
                                                                                               target_rec, target_f1))
        self.writer.close()
        return evaluation_results


def mse_loss(x, y):
    # mean squared error loss
    return torch.sum((x - y) ** 2) / x.data.shape[0]


def cosine_loss(x, y):
    c = nn.CosineSimilarity()
    return (1 - c(x, y)).mean()


def cos(x, y):
    """
    This returns the mean cosine similarity between two sets of vectors.
    """
    c = nn.CosineSimilarity()
    return c(x, y).mean()

def to_array(X, n=2):
    return np.array([np.eye(n)[x] for x in X])

def per_class_prec(y, pred):
    num_classes = len(set(y))
    y = to_array(y, num_classes)
    pred = to_array(pred, num_classes)
    results = []
    for j in range(num_classes):
        class_y = y[:, j]
        class_pred = pred[:, j]
        f1 = precision_score(class_y, class_pred, average='binary')
        results.append([f1])
    return np.array(results)

def per_class_rec(y, pred):
    num_classes = len(set(y))
    y = to_array(y, num_classes)
    pred = to_array(pred, num_classes)
    results = []
    for j in range(num_classes):
        class_y = y[:, j]
        class_pred = pred[:, j]
        f1 = recall_score(class_y, class_pred, average='binary')
        results.append([f1])
    return np.array(results)

def macro_f1(y, pred):
    """Get the macro f1 score"""

    num_classes = len(set(y))
    y = to_array(y, num_classes)
    pred = to_array(pred, num_classes)

    results = []
    for j in range(num_classes):
        class_y = y[:, j]
        class_pred = pred[:, j]
        f1 = f1_score(class_y, class_pred, average='binary')
        results.append([f1])
    return np.array(results).mean()

def get_best_run(weightdir):
    """
    This returns the best dev f1, parameters, and weights from the models
    found in the weightdir.
    """
    best_params = []
    best_f1 = 0.0
    best_weights = ''
    logger.info("Your directory for the weights is: "+weightdir)
    tempfilelist = os.listdir(weightdir)
    if ".DS_Store" in tempfilelist:
        tempfilelist.remove(".DS_Store")
    for file in tempfilelist:
        if os.path.isfile(os.path.join(weightdir,file)):
            epochs = int(re.findall('[0-9]+', file.split('-')[-4])[0])
            batch_size = int(re.findall('[0-9]+', file.split('-')[-3])[0])
            alpha = float(re.findall('0.[0-9]+', file.split('-')[-2])[0])
            f1 = float(re.findall('0.[0-9]+', file.split('-')[-1])[0])
            if f1 > best_f1:
                best_params = [epochs, batch_size, alpha]
                best_f1 = f1
                weights = os.path.join(weightdir, file)
                best_weights = weights
    return best_f1, best_params, best_weights

def get_last_run(weightdir):
    """
    Finds the file with the greatest number of epochs.
    Reminder: Delete all models before starting each experiment
    """
    tempfilelist = os.listdir(weightdir)
    greatest = 0
    last_run_file_name = ""
    logger.info("Your directory for the weights is: " + weightdir)
    if ".DS_Store" in tempfilelist:
        tempfilelist.remove(".DS_Store")
    for file in tempfilelist:
        if os.path.isfile(os.path.join(weightdir, file)):
            epochs = int(re.findall('[0-9]+', file.split('-')[-4])[0])
            if epochs > greatest:
                greatest = epochs
                last_run_file_name = file
    return os.path.abspath(last_run_file_name)

def read_config(config_path):
    with open(config_path, 'rb') as ymlfile:
        config = yaml.load(ymlfile, yaml.FullLoader)
        return config['tasks']

def set_parameters(task_config, task):
    argus = task_config.keys()
    logger.info("These are your specified arguments: ")
    print(argus)

    if 'concept' not in argus:
        logger.warning("No concept specified, please choose a concept!")
    else:
        concept = task_config['concept']

    if 'source_language' not in argus:
        task_config['source_language'] = 'en'  # default value
        logger.info("Source language set to default ({})".format(task_config['source_language']))

    if 'target_language' not in argus:
        task_config['target_language'] = 'fr'  # default value
        logger.info("Target language set to default ({})".format(task_config['target_language']))

    if 'epochs' not in argus:
        task_config['epochs'] = 200  # default value
        logger.info("Epochs set to default ({})".format(task_config['epochs']))

    if 'source_embeddings' not in argus:
        logger.warning("Source embeddings not found in config, please specify!")

    if 'target_embeddings' not in argus:
        logger.warning("Target embeddings not found in config, please specify!")

    if 'lexicon' not in argus:
        logger.warning("Lexicon not found in config, please specify!")

    lex = task_config['lexicon'].split("/")[-1].split(".")[0]

    embed_s = task_config['source_embeddings'].split("/")[-1].split(".")[0]

    embed_t = task_config['target_embeddings'].split("/")[-1].split(".")[0]

    dir_name = "_".join([concept, lex, embed_s, embed_t])

    if 'number_cat' not in argus:
        task_config['number_cat'] = 'binary'  # default value
        logger.info("Classification set to default ({})".format(task_config['number_cat']))

    if 'alpha' not in argus:
        task_config['alpha'] = .001  # default value
        logger.info("Alpha set to default ({})".format(task_config['alpha']))

    if 'proj_loss' not in argus:
        task_config['proj_loss'] = 'mse'  # default value
        logger.info("Projection loss set to default ({})".format(task_config['proj_loss']))

    if 'batch_size' not in argus:
        task_config['batch_size'] = 50  # default value
        logger.info("Batch size set to default ({})".format(task_config['batch_size']))

    if 'source_dataset' not in argus:
        ### default = 'datasets/...'
        logger.warning("No source dataset specified, please provide a source dataset!")

    if 'target_dataset' not in argus:
        ### default = 'datasets/...'
        logger.warning("No target dataset specified, please provide a target dataset!")

    if task == 'BL+LTTL':
        if 'bl1' in task_config['target_dataset'].lower():
            bl = 'BL1'
        elif 'bl2' in task_config['target_dataset'].lower():
            bl = 'BL2'
        else:
            logger.warning("No baseline nr could be detected, please specify one!")

        if 'models_dir' not in argus:
            logger.warning("No directory for trained models given! Please provide one!")

        if 'results_dir' not in argus:
            ### help='where to save the results for each run (default: ./results)',
            ### default='results')

            task_config['results_dir'] = os.path.join("./results", dir_name + '_combi_' + bl)
            logger.info("Directory for results set to default ({})".format(task_config['results_dir']))

        if 'runs_dir' not in argus:
            task_config['runs_dir'] = os.path.join("./runs", dir_name + '_combi_' + bl)
            logger.info("Directory for (tensorboard) runs set to default ({})".format(task_config['runs_dir']))

        if 'preds_dir' not in argus:
            task_config['preds_dir'] = os.path.join("./predictions", dir_name + '_combi_' + bl)
            logger.info("Directory for predictions set to default ({})".format(task_config['preds_dir']))

    if task == 'LTTL+BL':
        if 'models_dir' not in argus:
            logger.warning("No directory for trained models given! Please provide one!")

        if 'results_dir' not in argus:
            ### help='where to save the results for each run (default: ./results)',
            ### default='results')

            task_config['results_dir'] = os.path.join("./results", dir_name + '_combi')
            logger.info("Directory for results set to default ({})".format(task_config['results_dir']))

        if 'runs_dir' not in argus:
            task_config['runs_dir'] = os.path.join("./runs", dir_name + '_combi')
            logger.info("Directory for (tensorboard) runs set to default ({})".format(task_config['runs_dir']))

        if 'preds_dir' not in argus:
            task_config['preds_dir'] = os.path.join("./predictions", dir_name + '_combi')
            logger.info("Directory for predictions set to default ({})".format(task_config['preds_dir']))

    else:
        if 'models_dir' not in argus:
            ###  help="where to dump weights during training (default: ./models/blse)"

            task_config['models_dir'] = os.path.join("./models", dir_name)
            logger.info("Directory for saving models set to default ({})".format(task_config['models_dir']))

        if 'results_dir' not in argus:
            ### help='where to save the results for each run (default: ./results)',
            ### default='results')

            task_config['results_dir'] = os.path.join("./results", dir_name)
            logger.info("Directory for results set to default ({})".format(task_config['results_dir']))

        if 'runs_dir' not in argus:
            task_config['runs_dir'] = os.path.join("./runs", dir_name)
            logger.info("Directory for (tensorboard) runs set to default ({})".format(task_config['runs_dir']))

        if 'preds_dir' not in argus:
            task_config['preds_dir'] = os.path.join("./predictions", dir_name)
            logger.info("Directory for predictions set to default ({})".format(task_config['preds_dir']))

    # make dirs if they are not there yet
    os.makedirs(task_config['results_dir'], exist_ok=True)
    os.makedirs(task_config['preds_dir'], exist_ok=True)
    os.makedirs(task_config['runs_dir'], exist_ok=True)
    if (task == 'LTTL+BL') or (task == 'BL+LTTL'):
        if not os.path.isdir(task_config['models_dir']):
            logger.warning("Chosen model directory does not exist! Please chose another one!")
    else:
        os.makedirs(task_config['models_dir'], exist_ok=True)

    return task_config


def lttl(task_config, task):
    set_task_config = set_parameters(task_config, task)

    # saves the configurations given in the arg parse to json to create the results file
    variables = json.dumps(set_task_config, indent=2)

    alpha = set_task_config['alpha']
    batch_size = set_task_config['batch_size']
    num_cat = set_task_config['number_cat']
    epochs = set_task_config['epochs']
    savedir = set_task_config['models_dir']
    runsdir = set_task_config['runs_dir']
    resultsdir = set_task_config['results_dir']
    predsdir = set_task_config['preds_dir']

    # import datasets (representation will depend on final classifier)
    logger.info('importing datasets')

    dataset = CorpusDataset(os.path.join('datasets', set_task_config['source_dataset']),
                            None,
                            number_cat=set_task_config['number_cat'],
                            rep=words,
                            one_hot=False)

    cross_dataset = CorpusDataset(os.path.join('datasets', set_task_config['target_dataset']),
                                  None,
                                  number_cat=set_task_config['number_cat'],
                                  rep=words,
                                  one_hot=False,
                                  train_set=False)

    # Import monolingual vectors
    logger.info('importing word embeddings')
    src_vecs = WordVecs(set_task_config['source_embeddings'])
    logger.info(f" The source vecs length is {src_vecs.vector_size}")
    trg_vecs = WordVecs(set_task_config['target_embeddings'])
    logger.info(f" The target vecs length is {trg_vecs.vector_size}")

    # Import translation pairs
    logger.info('importing translation pairs')
    pdataset = ProjectionDataset(set_task_config['lexicon'], src_vecs, trg_vecs)

    if num_cat == "binary":
        output_dim = 2
        b = 'bi'
    elif num_cat == "three class":
        output_dim = 3
        b = '3cls'
    elif num_cat == "four class":
        output_dim = 4
        b = '4cls'

    # Create a BLSE object
    blse = LTTL(src_vecs, trg_vecs, pdataset=pdataset, cdataset=dataset, trg_dataset=cross_dataset,
                projection_loss=set_task_config['proj_loss'],
                output_dim=output_dim,
                summary_writter_params=runsdir
                )

    # Fit the model
    blse.fit(pdataset._Xtrain, pdataset._ytrain,
             dataset._Xtrain, dataset._ytrain,
             weight_dir=savedir,
             batch_size=batch_size, alpha=alpha, epochs=epochs)

    # Get last run (results for last run are in results file)
    path_last_run = json.dumps(os.path.abspath(get_last_run(savedir)))

    # Get best run
    best_f1, best_params, best_weights = get_best_run(savedir)
    try:
        blse.load_weights(best_weights)
        #print(best_weights)
        print('Dev set')
        print('best dev f1: {0:.3f}'.format(best_f1))
        print('parameters: epochs {0} batch size {1} alpha {2}'.format(*best_params))
    except RuntimeError:
        logger.info("Get best_run was unsucessful. "
                    "The embeddings used in the experiment have different dimensions from the ones saved in models.")

    best_run_path = 'epochs{0}-batchsize{1}-alpha{2}'.format(*best_params) + "-f1{0:0.3f}".format(best_f1)

    # The test set in the target language is used to evaluate the model
    blse.evaluate(cross_dataset._Xtest, cross_dataset._ytest, src=False, outfile=os.path.join(predsdir,
                                                                                              '{0}-{1}-alpha{2}-epoch{3}-batch{4}.txt'.format(
                                                                                                  'preds', b, alpha,
                                                                                                  best_params[0],
                                                                                                  batch_size)))

    blse.confusion_matrix(cross_dataset._Xtest, cross_dataset._ytest, src=False)
    results = blse.evaluation_results()
    test_results = blse.evaluate(cross_dataset._Xtest, cross_dataset._ytest, src=False)

    def json_meta_information():
        '''Creates json-file with configurations and results in one file'''
        meta = {"time_stamp": str(run_time),
                "run_file": os.fspath(runsdir), "hyperparameters": variables,
                "model": {"vector_info": json.loads(blse.vector_info), "lexicon_size": str(pdataset.info),
                          "lexicon_coverage": json.loads(blse.look_up_meta_info_for_results(pdataset._Xtrain, pdataset._ytrain)),
                          "model_path (last_run)": json.loads(path_last_run), "model_best_run_path": best_run_path,
                          "corpora_info": {"source_dataset": dataset.info, "target_dataset": cross_dataset.info}},
                "results": json.loads(results),
                "test results": json.loads(test_results)}
        return meta

    experiment_hyperparams = ("alpha " + str(alpha) + " epochs " + str(epochs) + " batch_size " + str(batch_size))
    file_name = "".join(run_time.strftime("%d-%m-%Y %H:%M:%S") + " " + experiment_hyperparams) ##run_time_now??

    with open(os.path.join(resultsdir, file_name + ".json"), "w") as result_file:
        meta = json_meta_information()
        json.dump(meta, result_file, indent=2)

    #print(json_meta_information())


def combi(task_config, task):
    set_task_config = set_parameters(task_config, task)

    alpha = set_task_config['alpha']
    batch_size = set_task_config['batch_size']
    num_cat = set_task_config['number_cat']
    savedir = set_task_config['models_dir']
    predsdir = set_task_config['preds_dir']

    # import datasets (representation will depend on final classifier)
    logger.info('importing datasets')

    cross_dataset = CorpusDataset(os.path.join('datasets', set_task_config['target_dataset']),
                                  None,
                                  number_cat=set_task_config['number_cat'],
                                  rep=words,
                                  one_hot=False,
                                  train_set=False, combination=True)

    # Import monolingual vectors
    logger.info('importing word embeddings')
    src_vecs = WordVecs(set_task_config['source_embeddings'])
    logger.info(f" The source vecs length is {src_vecs.vector_size}")
    trg_vecs = WordVecs(set_task_config['target_embeddings'])
    logger.info(f" The target vecs length is {trg_vecs.vector_size}")

    if num_cat == "binary":
        output_dim = 2
        b = 'bi'
    elif num_cat == "three class":
        output_dim = 3
        b = '3cls'
    elif num_cat == "four class":
        output_dim = 4
        b = '4cls'

    # Create a BLSE object
    blse = LTTL(src_vecs, trg_vecs, cross_dataset, output_dim=output_dim)

    # Get best run
    best_f1, best_params, best_weights = get_best_run(savedir)
    try:
        blse.load_weights(best_weights)
        #print(best_weights)
    except RuntimeError:
        logger.info("Get best_run was unsucessful. "
                    "The embeddings used in the experiment have different dimensions from the ones saved in models.")

    blse.evaluation_preds(cross_dataset._Xtest, cross_dataset._docid_test, outfile=os.path.join(predsdir,
                                                                                                '{0}-{1}-alpha{2}-epoch{3}-batch{4}.txt'.format(
                                                                                                    'preds', b, alpha,
                                                                                                    best_params[0],
                                                                                                    batch_size)),
                                                                                                    src=False)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cf', '--config_file',
                        help="path to configuration file",
                        default='config_test.yaml')

    args = parser.parse_args()

    config = read_config(args.config_file)
    for t in config:
        task = list(t)[0]
        arguments = t[task]
        if task == 'LTTL':
            lttl(arguments, task)
        elif task == 'BL+LTTL' or 'LTTL+BL':
            combi(arguments, task)
        else:
            logger.warning("Mode incorrect - please choose between LTTL, LTTL+BL and BL+LTTL")


if __name__ == '__main__':
    main()



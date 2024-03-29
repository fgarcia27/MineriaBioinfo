# -*- coding: UTF-8 -*-

import os
from itertools import chain
from optparse import OptionParser
from time import time
from collections import Counter
import re

import nltk
import sklearn
import scipy.stats
import sys

from sklearn.externals import joblib
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

from nltk.corpus import stopwords


# Objective
# Training and evaluation of CRFs with sklearn-crfsuite.
#
# Input parameters
# --inputPath=PATH    Path of training and test data set
# --trainingFile        File with training data set
# --testFile        File with test data set
# --outputPath=PATH    Output path to place output files
# --filteringStopWords   Filtering stop words
# --excludeSymbols      Filtering punctuation marks

# Output
# 1) Best model

# Examples
# python training-validation-v1.py
# --inputPath /home/compu2/bionlp/conditional-random-fields/data-sets
# --trainingFile training-data-set-70.txt
# --testFile test-data-set-30.txt
# --outputPath /home/compu2/bionlp/conditional-random-fields
# python training-validation-v1.py --inputPath /home/compu2/bionlp/conditional-random-fields/data-sets --trainingFile training-data-set-70.txt --testFile test-data-set-30.txt --outputPath /home/compu2/bionlp/conditional-random-fields

#################################
#           FUNCTIONS           #
#################################

def word2features(sent, i):
    listElem = sent[i].split('|')
    word = listElem[0]
    #print("word: {}".format(word))
    lemma = listElem[1]
    postag = listElem[2]

    features = {
        #'word': word,
        'lemma': lemma,
        'postag': postag,
    }
    if i > 0:
        listElem = sent[i - 1].split('|')
        #word1 = listElem[0]
        lemma1 = listElem[1]
        postag1 = listElem[2]
        features.update({
            #'-1:word': word1,
            '-1:lemma': lemma1,
            '-1:postag': postag1,
        })

    if i < len(sent) - 1:
        listElem = sent[i + 1].split('|')
        #word1 = listElem[0]
        lemma1 = listElem[1]
        postag1 = listElem[2]
        features.update({
            #'+1:word': word1,
            '+1:lemma': lemma1,
            '+1:postag': postag1,
        })
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [elem.split('|')[3] for elem in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def print_transitions(trans_features, f):
    for (label_from, label_to), weight in trans_features:
        f.write("{:6} -> {:7} {:0.6f}\n".format(label_from, label_to, weight))


def print_state_features(state_features, f):
    for (attr, label), weight in state_features:
        f.write("{:0.6f} {:8} {}\n".format(weight, label, attr.encode("utf-8")))


__author__ = 'CMendezC'

##########################################
#               MAIN PROGRAM             #
##########################################

if __name__ == "__main__":
    # Defining parameters
    parser = OptionParser()
    parser.add_option("--inputPath", dest="inputPath",
                      help="Path of training data set", metavar="PATH")
    parser.add_option("--outputPath", dest="outputPath",
                      help="Output path to place output files",
                      metavar="PATH")
    parser.add_option("--trainingFile", dest="trainingFile",
                      help="File with training data set", metavar="FILE")
    parser.add_option("--testFile", dest="testFile",
                      help="File with test data set", metavar="FILE")
    parser.add_option("--excludeStopWords", default=False,
                      action="store_true", dest="excludeStopWords",
                      help="Exclude stop words")
    parser.add_option("--excludeSymbols", default=False,
                      action="store_true", dest="excludeSymbols",
                      help="Exclude punctuation marks")

    (options, args) = parser.parse_args()
    if len(args) > 0:
        parser.error("Any parameter given.")
        sys.exit(1)

    print('-------------------------------- PARAMETERS --------------------------------')
    print("Path of training data set: " + options.inputPath)
    print("File with training data set: " + str(options.trainingFile))
    print("Path of test data set: " + options.inputPath)
    print("File with test data set: " + str(options.testFile))
    print("Exclude stop words: " + str(options.excludeStopWords))
    symbols = ['.', ',', ':', ';', '?', '!', '\'', '"', '<', '>', '(', ')', '-', '_', '/', '\\', '¿', '¡', '+', '{',
               '}', '[', ']', '*', '%', '$', '#', '&', '°', '`', '...']
    #print("Exclude symbols " + str(symbols) + ': ' + str(options.excludeSymbols))
    print("Exclude symbols: " + str(options.excludeSymbols))

    print('-------------------------------- PROCESSING --------------------------------')
    print('Reading corpus...')
    t0 = time()

    sentencesTrainingData = []
    sentencesTestData = []

    stopwords = [word for word in stopwords.words('english')]

    with open(os.path.join(options.inputPath, options.trainingFile), "r") as iFile:
        for line in iFile.readlines():
            listLine = []
            line = line.strip('\n')
            for token in line.split():
                if options.excludeStopWords:
                    listToken = token.split('|')
                    lemma = listToken[1]
                    if lemma in stopwords:
                        continue
                if options.excludeSymbols:
                    listToken = token.split('|')
                    lemma = listToken[1]
                    if lemma in symbols:
                        continue
                listLine.append(token)
            sentencesTrainingData.append(listLine)
        print("   Sentences training data: " + str(len(sentencesTrainingData)))

    with open(os.path.join(options.inputPath, options.testFile), "r") as iFile:
        for line in iFile.readlines():
            listLine = []
            line = line.strip('\n')
            for token in line.split():
                if options.excludeStopWords:
                    listToken = token.split('|')
                    lemma = listToken[1]
                    if lemma in stopwords:
                        continue
                if options.excludeSymbols:
                    listToken = token.split('|')
                    lemma = listToken[1]
                    if lemma in symbols:
                        continue
                listLine.append(token)
            sentencesTestData.append(listLine)
        print("   Sentences test data: " + str(len(sentencesTestData)))

    print("Reading corpus done in: %fs" % (time() - t0))

    #print(sent2features(sentencesTrainingData[0])[0])
    #print(sent2features(sentencesTestData[0])[0])
    t0 = time()

    X_train = [sent2features(s) for s in sentencesTrainingData]
    print(X_train[0])
    y_train = [sent2labels(s) for s in sentencesTrainingData]
    print(y_train[0])

    X_test = [sent2features(s) for s in sentencesTestData]
    # print X_test
    y_test = [sent2labels(s) for s in sentencesTestData]

    # Fixed parameters
    crf = sklearn_crfsuite.CRF(
         algorithm='lbfgs',
         #c1=0.1,
         #c2=0.1,
         max_iterations=100
         #,
         #all_possible_transitions=True
     )

    # Original: labels = list(crf.classes_)
    # Original: labels.remove('O')
    labels = list(['GENE'])

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted', labels=labels)

    # Fixed parameters
    crf.fit(X_train, y_train)

    # Best hiperparameters
    # crf = rs.best_estimator_
    nameReport = options.trainingFile.replace('.txt', '.fStopWords_' + str(options.excludeStopWords) + '.fSymbols_' + str(
        options.excludeSymbols) + '.txt')
    with open(os.path.join(options.outputPath, "reports", "report_" + nameReport), mode="w") as oFile:
        oFile.write("********** TRAINING **********\n")
        oFile.write("Training file: " + options.trainingFile + '\n')
        oFile.write('\n')
        oFile.write("Model: {}".format(crf))

    print("Training done in: %fs" % (time() - t0))
    t0 = time()

    # Saving model
    print("     Saving training model...")
    t1 = time()
    nameModel = options.trainingFile.replace('.txt', '.fStopWords_' + str(options.excludeStopWords) + '.fSymbols_' + str(
        options.excludeSymbols) + '.mod')
    joblib.dump(crf, os.path.join(options.outputPath, "models", nameModel))
    print("        Saving training model done in: %fs" % (time() - t1))

    # Evaluation against evaluation data
    y_pred = crf.predict(X_test)
    print("*********************************")
    name = options.trainingFile.replace('.txt', '.fStopWords_' + str(options.excludeStopWords) + '.fSymbols_' + str(
        options.excludeSymbols) + '.txt')
    with open(os.path.join(options.outputPath, "reports", "y_pred_" + name), "w") as oFile:
        for y in y_pred:
            oFile.write(str(y) + '\n')

    print("*********************************")
    name = options.trainingFile.replace('.txt', '.fStopWords_' + str(options.excludeStopWords) + '.fSymbols_' + str(
        options.excludeSymbols) + '.txt')
    with open(os.path.join(options.outputPath, "reports", "y_test_" + name), "w") as oFile:
        for y in y_test:
            oFile.write(str(y) + '\n')

    print("Prediction done in: %fs" % (time() - t0))

    # labels = list(crf.classes_)
    # labels.remove('O')

    with open(os.path.join(options.outputPath, "reports", "report_" + nameReport), mode="a") as oFile:
        oFile.write('\n********** EVALUATION **********\n')
        oFile.write("Flat F1: " + str(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)))
        oFile.write('\n')
        # labels = list(crf.classes_)
        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )
        oFile.write(metrics.flat_classification_report(
            y_test, y_pred, labels=sorted_labels, digits=3
        ))
        oFile.write('\n')

        oFile.write("\nTop likely transitions:\n")
        print_transitions(Counter(crf.transition_features_).most_common(50), oFile)
        oFile.write('\n')

        oFile.write("\nTop unlikely transitions:\n")
        print_transitions(Counter(crf.transition_features_).most_common()[-50:], oFile)
        oFile.write('\n')

        oFile.write("\nTop positive:\n")
        print_state_features(Counter(crf.state_features_).most_common(200), oFile)
        oFile.write('\n')

        oFile.write("\nTop negative:\n")
        print_state_features(Counter(crf.state_features_).most_common()[-200:], oFile)
        oFile.write('\n')

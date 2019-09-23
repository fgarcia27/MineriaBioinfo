# -*- coding: UTF-8 -*-

import os
from itertools import chain
from optparse import OptionParser
from time import time
from collections import Counter

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
#from trainingTesting_Sklearn_crfsuite import word2features
#from training_validation_v2 import sent2features
# from trainingTesting_Sklearn_crfsuite import hasNonAlphaNum
# from trainingTesting_Sklearn_crfsuite import hasDigit

# Objective
# Tagging transformed file with CRF model with sklearn-crfsuite.
#
# Input parameters
# --inputPath=PATH    Path of transformed files x|y|z
# --modelPath        Path to CRF model
# --modelName    Model name
# --outputPath=PATH    Output path to place output files
# --filteringStopWords   Filtering stop words
# --filterSymbols      Filtering punctuation marks

# Output
# 1) Tagged files in transformed format

# Examples
# Sentences
# C:\Anaconda2\python tagging_Sklearn_crfsuite.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\classifying_TFSentences\corpus\ECK120011394_FhlA\transformed --modelName aspectsTraining.fStopWords_False.fSymbols_True --modelPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\classifying_TFSentences\corpus\ECK120011394_FhlA\transformed_CRFtagged --filterSymbols > output.taggingCRF.20161107.txt
# C:\Anaconda2\python tagging_Sklearn_crfsuite.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\classifying_TFSentences\corpus\ECK120011394_FhlA\transformed --modelName sentencesTraining.fStopWords_False.fSymbols_False --modelPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\classifying_TFSentences\corpus\ECK120011394_FhlA\transformed_CRFtagged > output.taggingCRF.20161107.txt
# python
# tagging_Sklearn_crfsuite.py
# --inputPath /home/compu2/bionlp/conditional-random-fields/input
# --modelName training-data-set-70.fStopWords_False.fSymbols_False
# --modelPath /home/compu2/bionlp/conditional-random-fields
# --outputPath /home/compu2/bionlp/conditional-random-fields
# python tagging_Sklearn_crfsuite.py --inputPath /home/compu2/bionlp/conditional-random-fields/input --modelName training-data-set-70.fStopWords_False.fSymbols_False --modelPath /home/compu2/bionlp/conditional-random-fields --outputPath /home/compu2/bionlp/conditional-random-fields

#################################
#           FUNCTIONS           #
#################################
def word2features(sent, i):
    listElem = sent[i].split('|')
    word = listElem[0]
    lemma = listElem[1]
    postag = listElem[2]

    features = {
        # Suffixes
        #'word[-3:]': word[-3:],
        #'word[-2:]': word[-2:],
        #'word[-1:]': word[-1:],
        #'word.isupper()': word.isupper(),
        #'word': word,
        #'lemma': lemma,
        #'postag': postag,
        'lemma[-3:]': lemma[-3:],
        'lemma[-2:]': lemma[-2:],
        'lemma[-1:]': lemma[-1:],
        'lemma[+3:]': lemma[:3],
        'lemma[+2:]': lemma[:2],
        'lemma[+1:]': lemma[:1],
        #'word[:3]': word[:3],
        #'word[:2]': word[:2],
        #'word[:1]': word[:1],
        #'endsConLow()={}'.format(endsConLow(word)): endsConLow(word),
    }
    if i > 0:
        listElem = sent[i - 1].split('|')
        word1 = listElem[0]
        lemma1 = listElem[1]
        postag1 = listElem[2]
        features.update({
            #'-1:word': word1,
            '-1:lemma': lemma1,
            '-1:postag': postag1,
        })

    if i < len(sent) - 1:
        listElem = sent[i + 1].split('|')
        word1 = listElem[0]
        lemma1 = listElem[1]
        postag1 = listElem[2]
        features.update({
            #'+1:word': word1,
            '+1:lemma': lemma1,
            '+1:postag': postag1,
        })

    '''    
    if i > 1:
        listElem = sent[i - 2].split('|')
        word2 = listElem[0]
        lemma2 = listElem[1]
        postag2 = listElem[2]
        features.update({
            '-2:word': word2,
            '-2:lemma': lemma2,
        })

    if i < len(sent) - 2:
        listElem = sent[i + 2].split('|')
        word2 = listElem[0]
        lemma2 = listElem[1]
        postag2 = listElem[2]
        features.update({
            '+2:word': word2,
            '+2:lemma': lemma2,
        })

    trigrams = False
    if trigrams:
        if i > 2:
            listElem = sent[i - 3].split('|')
            word3 = listElem[0]
            lemma3 = listElem[1]
            postag3 = listElem[2]
            features.update({
                '-3:word': word3,
                '-3:lemma': lemma3,
            })

        if i < len(sent) - 3:
            listElem = sent[i + 3].split('|')
            word3 = listElem[0]
            lemma3 = listElem[1]
            postag3 = listElem[2]
            features.update({
                '+3:word': word3,
                '+3:lemma': lemma3,
            })
    '''
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
    parser.add_option("--modelPath", dest="modelPath",
                      help="Path to read CRF model",
                      metavar="PATH")
    parser.add_option("--modelName", dest="modelName",
                      help="Model name", metavar="TEXT")
    parser.add_option("--filterStopWords", default=False,
                      action="store_true", dest="filterStopWords",
                      help="Filtering stop words")
    parser.add_option("--filterSymbols", default=False,
                      action="store_true", dest="filterSymbols",
                      help="Filtering punctuation marks")

    (options, args) = parser.parse_args()
    if len(args) > 0:
        parser.error("Any parameter given.")
        sys.exit(1)

    print('-------------------------------- PARAMETERS --------------------------------')
    print("Path to read input files: " + options.inputPath)
    print("Mode name: " + str(options.modelName))
    print("Model path: " + options.modelPath)
    print("Path to place output files: " + options.outputPath)
    print("Filtering stop words: " + str(options.filterStopWords))
    symbols = ['.', ',', ':', ';', '?', '!', '\'', '"', '<', '>', '(', ')', '-', '_', '/', '\\', '¿', '¡', '+', '{',
               '}', '[', ']', '*', '%', '$', '#', '&', '°', '`', '...']
    # symbols = [sym.decode('utf-8') for sym in ['.', ',', ':', ';', '?', '!', '\'', '"', '<', '>', '(', ')', '-', '_', '/', '\\', '¿', '¡', '+', '{',
    #            '}', '[', ']', '*', '%', '$', '#', '&', '°']]
    # symbols = [u'.', u',', u':', u';', u'?', u'!', u'\'', u'"', u'<', u'>', u'(', u')', u'-', u'_', u'/', u'\\', u'¿', u'¡', u'+', u'{',
    #             u'}', u'[', u']', u'*', u'%', u'$', u'#', u'&', u'°', u'`']
    print("Filtering symbols " + str(symbols) + ': ' + str(options.filterSymbols))

    print('-------------------------------- PROCESSING --------------------------------')

    #stopwords = [word.decode('utf-8') for word in stopwords.words('english')]

    # Read CRF model
    t0 = time()
    print('Reading CRF model...')
    crf = joblib.load(os.path.join(options.modelPath, 'models', options.modelName + '.mod'))
    print("Reading CRF model done in: %fs" % (time() - t0))

    print('Processing corpus...')
    t0 = time()
    # labels = list(['MF', 'TF', 'DFAM', 'DMOT', 'DPOS', 'PRO'])
    # Walk directory to read files
    for path, dirs, files in os.walk(options.inputPath):
        # For each file in dir
        for file in files:
            print("   Preprocessing file..." + str(file))
            sentencesInputData = []
            sentencesOutputData = []
            with open(os.path.join(options.inputPath, file), "r") as iFile:
                lines = iFile.readlines()
                for line in lines:
                    listLine = []
                    # line = line.decode("utf-8")
                    for token in line.strip('\n').split():
                        if options.filterStopWords:
                            listToken = token.split('|')
                            lemma = listToken[1]
                            # Original if lemma in stopwords.words('english'):
                            if lemma in stopwords:
                                continue
                        if options.filterSymbols:
                            listToken = token.split('|')
                            lemma = listToken[1]
                            if lemma in symbols:
                                if lemma == ',':
                                    print("Coma , identificada")
                                continue
                        listLine.append(token)
                    sentencesInputData.append(listLine)
                print("   Sentences input data: " + str(len(sentencesInputData)))
                # print sentencesInputData[0]
                # print(sent2features(sentencesInputData[0])[0])
                # print(sent2labels(sentencesInputData[0]))
                X_input = [sent2features(s) for s in sentencesInputData]
                print(sent2features(sentencesInputData[0])[0])
                # y_test = [sent2labels(s) for s in sentencesInputData]
                # Predicting tags
                t1 = time()
                print("   Predicting tags with model")
                y_pred = crf.predict(X_input)
                #print y_pred[0]
                print("      Prediction done in: %fs" % (time() - t1))
                exit

                # Tagging with CRF model
                print("   Tagging file")
                for line, tagLine in zip(lines, y_pred):
                    outputLine = ''
                    idx_tagLine = 0
                    line = line.strip('\n')
                    print("\nLine: " + str(line))
                    print ("CRF tagged line: " + str(tagLine))
                    for token in line.split():
                        listToken = token.split('|')
                        word = listToken[0]
                        lemma = listToken[1]
                        tag = listToken[2]
                        if options.filterStopWords:
                            if lemma in stopwords:
                                outputLine += token + ' '
                                continue
                        if options.filterSymbols:
                            if lemma in symbols:
                                if lemma == ',':
                                    print("Coma , identificada")
                                outputLine += token + ' '
                                continue
                        CRFtag = tagLine[idx_tagLine]
                        #if (tag not in labels) and (CRFtag != 'O'):
                        #    print "*** CRF change token {} to {}".format(token, CRFtag)
                        #    outputLine += word + '|' + lemma + '|' + CRFtag + ' '
                        #else:
                        #    outputLine += word + '|' + lemma + '|' + tag + ' '
                        #idx_tagLine += 1
                    sentencesOutputData.append(outputLine.rstrip())
            with open(os.path.join(options.outputPath, file), "w") as oFile:
                for line in sentencesOutputData:
                    oFile.write(line + '\n')

    print("Processing corpus done in: %fs" % (time() - t0))

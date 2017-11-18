import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

from sklearn.model_selection import train_test_split

from feature_util_class import Selector, chiSquare, featureSelector
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score


# from feature_util_backup import * # X_train, X_test are from here. 
# from feature_util_backup import * 


# from classify_util import *

def testingMYhypothesis():
    word = "grain"

    lhs = sklearn_tfidf.vocabulary_[word]
    rhs = features_names.index(word)


    print("lhs and rhs are", lhs == rhs)


def getImportance(kbest):
    mask = kbest.get_support()
    np_fname = np.array(features_names)
    impName = np_fname[mask]
    impScore = kbest.scores_[mask]

    return sorted(zip(impScore, impName), reverse=True)



def comparison():

    chiS, chiN = zip(*getImportance(kbest1))
    imptfS, imptfN = zip(*getImportance(kbest2))

    chiF = list(termFrequencyCorpus[n] for n in chiN[:20])    
    imptfF = list(termFrequencyCorpus[n] for n in imptfN[:20])

    print(pd.DataFrame({"name":chiN[:20], "chiFreq":chiF}))
    print(pd.DataFrame({"name":imptfN[:20], "imptfF":imptfF}))



def classify(classifier, X_train, X_test, y_train, y_test):
    if classifier.lower() == "svm":
        clf = LinearSVC(random_state = 0)
    elif classifier.lower() == "bayes":
        clf = MultinomialNB()

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    f1_macro = f1_score(y_test, y_pred, average = 'macro')
    f1_micro = f1_score(y_test, y_pred, average = 'micro')
    f1_average = f1_score(y_test, y_pred, average = 'weighted')

    return clf, f1_macro, f1_micro, f1_average


def plotAllcombi():
    plt.figure()
    plt.plot(num_features, chi, 'o-', label="chiSquare")
    plt.plot(num_features, dfs, 'o-', label="DFS")
    plt.plot(num_features, gini, 'o-', label="GiniDFS-1")
    plt.plot(num_features, ginidfs, 'o-', label="GiniDFS-2")
    plt.plot(num_features, giniimptf, 'o-', label="GiniImpTfDFS")
    plt.legend(loc=0)
    plt.xlabel("Number of features")
    plt.xticks(num_features)
    plt.ylabel("f1-micro score")
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1,ncol=2, mode="expand", borderaxespad=0.)

    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title("comparsion of selector methods with varying number of features.")
    plt.show()



def plotClusterAnalysis(macro, micro, average):

    _, ax = plt.subplots()
    x = list(range((len(macro))))
    # print(x, len(macro), macro)
    ax.plot(x, macro, "o-", label="macro")
    ax.plot(x, micro, "o-", label="micro")
    ax.plot(x, average, "o-", label="average")

    a = ax.get_xticks().tolist()
    i = 0
    ax.set_xticks(x)

    print(clusters, threshold)
    for N_CLUSTERS in clusters:
        for THRESHOLD in threshold:
            print(N_CLUSTERS, THRESHOLD)
            a[i] = "(k=" + str(N_CLUSTERS)+ " th=" + str(THRESHOLD) + ")"
            i += 1
    
    ax.set_xticklabels(a, rotation='30')
    ax.set_xlabel("(cluster,threshold)")
    ax.set_ylabel("f1-Scores")
    ax.set_title("cluster analysis with "+str(ext)+" method")
    ax.legend()

    plt.show()



if __name__ == "__main__":


    N_CLUSTERS = 100
    THRESHOLD  = 0.9
    USE_KMEANS = 1
    NUM_FEATURES = 200

    clusters = [15,25,35]#list(range(10, 100, 10))
    threshold = [0.92]#, 0.95]#, 0.7, 0.8, 0.9, 1.0]
    num_features = [400, 500, 600, 700, 800, 900, 1000]
    selector_type = ["giniimptf", "dfs", "ginidfs", "chi", 'gini']

    macro = []
    micro = []
    average = []

    gini =[]; dfs=[]; ginidfs=[]; giniimptf= []; chi=[]


    # **** NOTE: for all comparison plot uncomment these two and comment two for loop just below.
    # for ext in selector_type:
    #     for NUM_FEATURES in num_features:

    # ***** NOTE: for these two for loops mention "ext" variable(values of selector_type), 
    #                                   for which cluster cleaning analysis is to be done.

    ext = "chi"
    for N_CLUSTERS in clusters:
        for THRESHOLD in threshold:

            Fobject = Selector(None, N_CLUSTERS, THRESHOLD, USE_KMEANS=USE_KMEANS)

            X_train = Fobject.X_train
            y_train = Fobject.y_train
            X_test = Fobject.X_test
            y_test = Fobject.y_test

            sklearn_tfidf, X_train_TD, X_test_TD = Fobject.makeTermDocMatrix(X_train, X_test)
            features_names = sklearn_tfidf.get_feature_names()
            print("Got TF-IDF")

            if ext == "giniimptf":
                X_train, X_test,kbest2 = featureSelector(X_train_TD, X_test_TD, y_train, NUM_FEATURES,\
                                                 Fobject.dictGiniImpTF(features_names),sklearn_tfidf)
                clf, f1_macro, f1_micro, f1_average =  classify("svm", X_train, X_test, y_train, y_test)
                giniimptf.append(f1_micro)

            
            elif ext == "dfs":
                X_train, X_test,kbest2 = featureSelector(X_train_TD, X_test_TD, y_train, NUM_FEATURES,\
                                                Fobject.dictDFS(features_names),sklearn_tfidf)
                clf, f1_macro, f1_micro, f1_average =  classify("svm", X_train, X_test, y_train, y_test)
                dfs.append(f1_micro)


            elif ext == "gini":
                X_train, X_test,kbest2 = featureSelector(X_train_TD, X_test_TD, y_train, NUM_FEATURES,\
                                                Fobject.dictGini(features_names),sklearn_tfidf)
                clf, f1_macro, f1_micro, f1_average =  classify("svm", X_train, X_test, y_train, y_test)
                gini.append(f1_micro)
                


            elif ext == "chi":
                X_train, X_test,kbest2 = chiSquare(X_train_TD, X_test_TD, y_train, NUM_FEATURES)

                clf, f1_macro, f1_micro, f1_average =  classify("svm", X_train, X_test, y_train, y_test)
                chi.append(f1_micro)


            elif ext == "ginidfs":
                X_train, X_test,kbest2 = featureSelector(X_train_TD, X_test_TD, y_train, NUM_FEATURES,\
                                                Fobject.dictGiniDFS(features_names),sklearn_tfidf)

                clf, f1_macro, f1_micro, f1_average =  classify("svm", X_train, X_test, y_train, y_test)
                ginidfs.append(f1_average)

                
            macro.append(f1_macro)
            micro.append(f1_micro)
            average.append(f1_average)



# plt.plot([0], macro, "o-",label="macro")
# plt.show()



# print("till here.")
plotClusterAnalysis(macro, micro, average)

# use this if first two for loops are used.
#comparsion of selector methods with varying number of features.
# plotAllcombi()
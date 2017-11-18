import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from sklearn.cluster import KMeans
# from classify_util import *



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import numpy as np


class Selector(object):
	"""docstring for featureSelector"""
	def __init__(self, dataset, N_CLUSTERS, THRESHOLD, USE_KMEANS):

		df_train = pd.read_csv("r8-train-stemmed.txt",names=["c","d"], sep="\t")
		df_test = pd.read_csv("r8-test-stemmed.txt",names=["c","d"],sep="\t")

		self.USE_KMEANS = USE_KMEANS
		self.N_CLUSTERS = N_CLUSTERS
		self.THRESHOLD = THRESHOLD


		self.X_train = df_train.d.values.tolist()
		self.y_train = df_train.c.values.tolist()
		self.X_test = df_test.d.values.tolist()
		self.y_test = df_test.c.values.tolist()

		self.wordsTrain = None
		self.allwords_train = None
		self.allwords_test = None
		self.docs = None
		self.LEN_OF_CLASS = None
		self.TOTAL_DOCS = None

		self.classes = list(df_train.c.unique())
		self.allwords_train = []
		_ = df_train.d.map(lambda x: self.allwords_train.extend(x.split()))
		self.wordsTrain = list(set(self.allwords_train)) # distinct words in training.
		self.allwords_test = []
		_ = df_test.d.map(lambda x: self.allwords_test.extend(x.split()))
		self.wordstest = list(set(self.allwords_test)) # distinct words in testing.
		self.allWords = []
		self.allWords.extend(self.wordsTrain)
		self.allWords.extend(self.wordstest)
		self.allWords = set(self.allWords)

		self.idToDoc = df_train.d.values.tolist()
		self.docsId = { cl: df_train.loc[df_train.c == cl, ["d"]]["d"].index.tolist() for cl in self.classes	}

		self.kmeans_cleaning()

		self.frequencyOfTermInClass = self.getFrequency_json()

		self.termFrequencyCorpus = \
		{term: \
		sum(self.frequencyOfTermInClass[cl][term] for cl in self.classes) for term in self.allWords}

		self.dictM = {cl:self.M(cl) for cl in self.classes}


	def getFrequency_json(self, ):
		if "frequency.json" not in os.listdir():
			frequencyOfTermInClass = {cl:
							 {term:sum(map(lambda x:x.count(term), docs[cl])) for term in self.
							 allWords}
							 for cl in classes
							}

			with open('frequency.json', 'w') as f:
				json.dump(frequencyOfTermInClass, f)

		else:  
			print("Found the frequency.json.")
			with open('frequency.json', 'r') as f:
				frequencyOfTermInClass = json.load(f)

		return frequencyOfTermInClass


	def kmeans_cleaning(self,):
		if self.USE_KMEANS:
			tfidf_info, tdmatrix, _ = self.makeTermDocMatrix(self.X_train)
			# print(tdmatrix)
			kmeans = KMeans(self.N_CLUSTERS, n_jobs=-1)
			X_new = kmeans.fit_transform(tdmatrix)   # hope they don't mess with indices of training data.
			corressponding_dists_with_indices_not_messed_hopefully = [X_new[(i, x)] for i,x in enumerate(kmeans.labels_)]
			print("len of list", len(corressponding_dists_with_indices_not_messed_hopefully))
			print("max threshold,",max(corressponding_dists_with_indices_not_messed_hopefully),"\n",\
				"min threshold:",
				min(corressponding_dists_with_indices_not_messed_hopefully))
			# plt.hist(corressponding_dists_with_indices_not_messed_hopefully)
			# plt.show()	
			cleanedId = (np.where(np.array(corressponding_dists_with_indices_not_messed_hopefully) < self.THRESHOLD)[0]).tolist()

			for cl in self.classes:
				self.docsId[cl] = set(self.docsId[cl]).intersection(cleanedId) # updating docsId


			# need to update X_train, y_train also.

			# print(type(cleanedId))
			print("Number of new documents:", len(cleanedId))
			# print(cleanedId[:5])
			self.X_train = (np.array(self.X_train)[cleanedId]).tolist()
			self.y_train = (np.array(self.y_train)[cleanedId]).tolist()


		self.docs = { cl: [self.idToDoc[x] for x in self.docsId[cl]] for cl in self.classes }

		self.LEN_OF_CLASS = {}
		for cl in self.classes:
			self.LEN_OF_CLASS[cl] = len(self.docs[cl])

		self.TOTAL_DOCS = sum(len(self.docs[cl]) for cl in self.classes)



	def countDocsPerClass(class_ , term):
		"""class_ : string of class
			 term   : string of the term.
			 
			 counts number of documents where class is class_ and term occurs.
			 
		"""
		doclistC = docs[class_]
		return sum(map(lambda x: term in x, doclistC))

	def countDocsWithoutTermPerClass(class_ , term):
		"""class_ : string of class
			 term   : string of the term.
			 
			 counts number of documents where class is class_ and term does not occurs.
			 
		"""
			
		doclistC = self.docs[class_]
		return sum(map(lambda x: term not in x, doclistC))

	def countDocs(term):
		"""Iterate over all of the classes. 
		"""
		return sum(countDocsPerClass(cl, term) for cl in self.classes)

	def classGivenTerm(_class, term):
		""" 
		Computes P(Ci|t)
		"""
		denominator = countDocs(term)
		numerator = countDocsPerClass(_class, term)
		if (denominator == 0):
			return 0
		else:
			return numerator/denominator
			
	def termGivenNotClass(_class, term):
		"""Computes P(t|~Ci)
		"""
		denominator = docsNotInClass = self.TOTAL_DOCS - self.LEN_OF_CLASS[_class]
		
		#numerator = sum(map(lambda cl: countDocsPerClass(cl, term), filter(lambda x: x!=_class, self.classes)))
		numerator = sum([countDocsPerClass(cl, term) for cl in filter(lambda x: x!=_class, self.classes)])
		
		return numerator/denominator


	def termAbsentGivenClass(_class, term):
		""" Computes P(~t|Ci)
		"""
		
		numerator = countDocsWithoutTermPerClass(_class, term)
		denominator = self.LEN_OF_CLASS[_class]
		
		return numerator/denominator


	def DFSi(_class, term):
		""" term: string.
				DFS: Distinguishing Feature Selector
				DFSi(t) =        P(Ci|t) 
								 -----------------------
								 1 + P(t|~Ci) + P(~t|Ci)
								 
				DFS(t) = sum(DFSi(t))         
		"""

		return classGivenTerm(_class, term) / (1 + termGivenNotClass(_class, term) + \
												termAbsentGivenClass(_class, term))

	def DFS(term):
		""" term: string.
				DFS: Distinguishing Feature Selector
				DFSi(t) =        P(Ci|t) 
								 -----------------------
								 1 + P(t|~Ci) + P(~t|Ci)
								 
				DFS(t) = sum(DFSi(t))         
		"""
		
		return sum([DFSi(cl, term) for cl in self.classes])   


	def Gini_DFSi(_class, term):
		"""term : string.
			 _class: string.
			 
			 calculates below for a term and _class.
			 returns this DFS(_class, term)*frequencyOfTermInClass["class"]["term"]
		"""
		classTermDict = self.frequencyOfTermInClass.get(_class, False)
		if not classTermDict:
			assert False

		return DFSi(_class, term)*classTermDict.get(term, 0) # zero is the default .

	def Gini_DFS(term):
		"""term: string.
		
			 return sum(Gini_DFSi())
		"""
		return sum([Gini_DFSi(cl, term) for cl in self.classes])   


	def dictGini():

		allGini = {cl:{term:Gini_DFS(term) for term in self.allWords}

			for cl in self.classes}


	def Ginii(_class, term):
		"""term : string.
			 _class: string.
			 
			 calculates below for a term and _class.
			 returns this classGivenTerm(_class, term)*frequencyOfTermInClass["class"]["term"]
		"""
		classTermDict = self.frequencyOfTermInClass.get(_class, False)
		if not classTermDict:
			assert False



		return classGivenTerm(_class, term)*classTermDict.get(term, 0) # zero is the default .

	def Gini(term):
		"""term: string.
		
			 return sum(Ginii())
		"""


		return sum([Ginii(cl, term) for cl in self.classes])



	def dictGini(self, features):

		print("dictGini")

		if "gini.json" not in os.listdir():
			with open("gini.json", "w") as f:
				allGini = {term:Gini(term) for term in features}
				json.dump(allGini, f)

			return allGini

		else:

			with open("gini.json", "r") as f:
				return json.load(f)


	def dictDFS(self, features):

		print("DFS")


		if "dfs.json" not in os.listdir():
			with open("dfs.json", "w") as f:
				allGini = {term:DFS(term) for term in features}

				json.dump(allGini, f)

			return allGini

		else:

			with open("dfs.json", "r") as f:
				return json.load(f)


	def dictGiniDFS(self, features, normalise = False):

		print("dictGiniDFS")

		if "ginidfs.json" not in os.listdir():
			with open("ginidfs.json", "w") as f:
				allGini = {term:Gini_DFS(term) for term in features}

				json.dump(allGini, f)

		else:

			with open("ginidfs.json", "r") as f:
				allGini =  json.load(f)


		if normalise:
			allGiniNormalise = {term:allGini[term]/self.termFrequencyCorpus[term] for term in features}



		return allGiniNormalise if normalise else allGini





	def ATFi(_class, term):
		"""	_class: string
			term : string.

			average term frequency in _class for the term "term"

		""" 

		denominator = countDocsPerClass(_class, term)
		# if denominator == 0:
			# assert False

		return self.frequencyOfTermInClass[_class][term]/(denominator+1)


	def M(self, _class):
		"""
		L(d_class) / D
		D: distinct words in _class.
		L(d_class) : no. of words in documents of _class.
		"""

		numerator  = sum(map(lambda x: len(x.split()), self.docs[_class]))
		denominator = len(self.frequencyOfTermInClass[_class].keys())

		return numerator/denominator

	# self.dictM = {cl:M(cl) for cl in self.classes}

	def impTFi(_class, term):
		"""
		"""

		return self.frequencyOfTermInClass[_class][term]* ATFi(_class, term)/dictM[_class]



	def GiniImpTF(term):
		"""
		"""
		return sum([DFSi(cl, term)*impTFi(cl, term) for cl in self.classes])


	def dictGiniImpTF(self,features):

		print("dictImpGiniTF")

		if "giniimpdfs.json" not in os.listdir():
			# pool = Pool(processes=3)

			allGini = {term:GiniImpTF(term) for term in features}

			with open("giniimpdfs.json", "w") as f:
				json.dump(allGini, f)

		else:
			with open("giniimpdfs.json", "r") as f:
				allGini =  json.load(f)


		return  allGini


	def makeTermDocMatrix(self, X_train, X_test=None):
		sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True,
					)#max_df = 2000)#, tokenizer=tokenize)
		sklearn_representation = sklearn_tfidf.fit_transform(X_train)
		sklearn_representation_test = sklearn_tfidf.transform(X_test) if X_test!=None else None

		return sklearn_tfidf, sklearn_representation, sklearn_representation_test


def chiSquare(X_train, X_test, y, numberOfFeatures):
	kbest = SelectKBest(chi2, k=numberOfFeatures)
	X_new = kbest.fit_transform(X_train, y)
	X_test = kbest.transform(X_test)
	return X_new, X_test, kbest


def featureSelector(X_train, X_test, y, numberOfFeatures, extractor, sklearn_tfidf):
	"""
	returns X_train, X_test, kbest on the "extractor" method.
	"""
	def scores(X, y):
		"""function to be passed as first parameter to kbest thing.
		"""
		return np.array([extractor.get(x,-1) for x in sklearn_tfidf.get_feature_names()])

	kbest = SelectKBest(scores, k=numberOfFeatures)
	X_new = kbest.fit_transform(X_train, y)
	X_test = kbest.transform(X_test)
	return X_new, X_test, kbest



   
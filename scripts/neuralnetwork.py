from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from copy import deepcopy
from random import shuffle
from string import punctuation
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm

import gensim as gs
import numpy as np

pd.options.mode.chained_assignment = None
tokenizer = TweetTokenizer()
tqdm.pandas(desc="progress-bar")
LabeledSentence = gs.models.doc2vec.LabeledSentence

def splitTweet(tweet):
	try:
		myTweet = unicode(tweet.decode('utf-8').lower())
		brokenTokens = tokenizer.tokenize(tweet)
		#now filter out extra symbols like urls, hashtags, cashtags, and other user refs
		brokenTokens = filter(lambda tw: not tw.startswith('http'), brokenTokens)
		brokenTokens = filter(lambda tw: not tw.startswith('#'), brokenTokens)
		brokenTokens = filter(lambda tw: not tw.startswith('@'), brokenTokens)
		brokenTokens = filter(lambda tw: not tw.startswith('$'), brokenTokens)
		return brokenTokens
	except:
		return 'DECODINGERROREXCEPTION'

def cleanTweets(data):
	data['tokens'] = data['Tweet'].progress_map(splitTweet)
	data = data[data.tokens != 'DECODINGERROREXCEPTION']
	data.reset_index(inplace=True)
	data.drop('index', inplace=True, axis=1)
	return data

def loadDataset():
	inputData = pd.read_csv('../data/datasubset.csv')
	inputData.drop(['ItemID', 'Date', 'Query', 'User'], axis=1, inplace=True)
	inputData = inputData[inputData.Sentiment.isnull() == False]

	#convert sentiment to int 
	inputData['Sentiment'] = inputData['Sentiment'].map(int)

	inputData = inputData[inputData['Tweet'].isnull() == False]
	inputData.reset_index(inplace=True)
	inputData.drop('index', axis=1, inplace=True)

	#randomly shuffle so as not to get skewed straightforward data
	inputData = inputData.reindex(np.random.permutation(inputData.index))
	return inputData

def makeLabelizedSentences(data, label):
	result = []
	for i,t in tqdm(enumerate(data)):
		label = '%s_%d'%(label,i)
		result.append(LabeledSentence(t, [label]))
	return result

n=10419
inputData = loadDataset()
inputData = cleanTweets(inputData)
#separating data of 10,419 tweets to test and train
x_train, x_test, y_train, y_test = train_test_split(np.array(inputData.head(n).tokens),np.array(inputData.head(n).Sentiment), test_size=0.2)

x_train=makeLabelizedSentences(x_train, 'TRAIN')
x_test=makeLabelizedSentences(x_test, 'TEST')

#at this point, we have a list of tokens and a label. so we can build a word2vec model now
wordList = [x.words for x in tqdm(x_train)]
dimensions=200
corpus_count=len(wordList) #number of items in wordList is fine to specify according to documentation
iter=5 #5 is the default according to documentation

tweet_vector = Word2Vec(size=dimensions, min_count=10)
tweet_vector.build_vocab(wordList)
tweet_vector.train(wordList, total_examples=corpus_count, epochs=iter)

print(tweet_vector['good'])

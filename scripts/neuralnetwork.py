from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from copy import deepcopy
from random import shuffle
from string import punctuation
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.layers import Activation, Dense

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

def buildFormattedWordVector(wordTokens, totalSize):
	vector = np.zeros(totalSize).reshape((1, totalSize))
	count = 0
	for wT in wordTokens:
		try:
			vector += tweet_vector[wT].reshape((1, totalSize)) * tfidf_matrix[wT]
			count += 1
		except KeyError:
			continue
	if count != 0:
		vector = vector / count
	return vector


n=10419
inputData = loadDataset()
inputData = cleanTweets(inputData)
#separating data of 10,419 tweets to test and train
x_train, x_test, y_train, y_test = train_test_split(np.array(inputData.head(n).tokens),np.array(inputData.head(n).Sentiment), test_size=0.2)

x_train=makeLabelizedSentences(x_train, 'TRAIN')
x_test=makeLabelizedSentences(x_test, 'TEST')

#at this point, we have a list of tokens and a label. so we can build a word2vec model now
dimensions=200

tweet_vector = Word2Vec(size=dimensions, min_count=10)
tweet_vector.build_vocab([x.words for x in tqdm(x_train)])
tweet_vector.train([x.words for x in tqdm(x_train)], total_examples=tweet_vector.corpus_count, epochs=tweet_vector.iter)

#now that we have vector for words, we need to combine them to make sentence vectors.
#could sum the vectors and do a average, but based on a research paper, a better accuracy can be obtained
#using weighted average through the tf-idf document inverse matrix. Details are in the paper

vc = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
myMatrix = vc.fit_transform([x.words for x in x_train])
tfidf_matrix = dict(zip(vc.get_feature_names(), vc.idf_))
print('Finished building tf-idf matrix. Final Matrix Size is {}'.format(len(tfidf_matrix)))

#now we can build the word vector using this inverse freqeuncy matrix

train_vectors = [buildFormattedWordVector(z, dimensions) for z in tqdm(map(lambda x: x.words, x_train))]
train_vectors = np.concatenate(train_vectors)
train_vectors = scale(train_vectors)

test_vectors = [buildFormattedWordVector(z, dimensions) for z in tqdm(map(lambda x: x.words, x_test))]
test_vectors = np.concatenate(test_vectors)
test_vectors = scale(test_vectors)

#now we can feed these vectors to a neural network classifier for processing using Keras
derivedModel = Sequential()
derivedModel.add(Dense(32, activation='relu', input_dim=dimensions))
derivedModel.add(Dense(1, activation='sigmoid'))
derivedModel.compile(optimizer = 'rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

derivedModel.fit(train_vectors, y_train, epochs=10, batch_size=32, verbose=2)
accuracy = derivedModel.evaluate(test_vectors, y_test, batch_size=128, verbose=2)
print('Accuracy achieved was', accuracy[1])

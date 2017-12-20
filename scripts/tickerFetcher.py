import ner
import json
import urllib

from nltk.tokenize import TweetTokenizer
from pandas as pd
from random import shuffle
from tqdm import tqdm
import numpy as np

#this script should read those tweets one by one and do processing on each one

def loadData():
	data = pd.read_csv('../data/donald.csv')
	data.drop(['Type','Media_Type', 'Hashtags', 'Tweet_Id', 'Tweet_Url', 'twt


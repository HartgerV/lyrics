import numpy as np
import nltk
from nltk.corpus import stopwords
from pandas import HDFStore

unknownWords = []
embedding_size = 300
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))
embeddingStore = HDFStore('mini.h5')
embeddingTable = embeddingStore.get('/mat')

def get_embedding(token):
    return np.array(embeddingTable.loc[token].to_list())

def document_embedding(document):
    # tokenize and lowercase
    tokens = [w.lower() for w in nltk.word_tokenize(document)]
    # initialize embedding sum vector
    embedding_sum = np.array([0] * embedding_size)
    embedding_count = 0
    for token in tokens:
        if token not in unknownWords and token not in stopWords:
            try:
                embedding_sum += get_embedding('/c/en/' + token)
                embedding_count += 1
            except:
                unknownWords.append(token)
    lyric_embedding = embedding_sum / embedding_count
    return lyric_embedding
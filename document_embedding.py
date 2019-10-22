import numpy as np
import nltk
from nltk.corpus import stopwords
import pandas

unknown_words = []
embedding_size = 300
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
embedding_store = pandas.read_hdf('mini.h5')
embedding_table = embedding_store.get('/mat')

def get_embedding(token):
    return np.array(embedding_table.loc[token].to_list())

def document_embedding(document):
    # tokenize and lowercase
    tokens = [w.lower() for w in nltk.word_tokenize(document)]
    # initialize embedding sum vector
    embedding_sum = np.array([0] * embedding_size)
    embedding_count = 0
    for token in tokens:
        if token not in unknown_words and token not in stop_words:
            try:
                embedding_sum += get_embedding('/c/en/' + token)
                embedding_count += 1
            except:
                unknown_words.append(token)
    lyric_embedding = embedding_sum / embedding_count
    return lyric_embedding
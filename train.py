import pandas as pd
import time
import random
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec


text = pd.read_json("yelp_academic_dataset_tip.json/yelp_academic_dataset_tip.json", lines=True)["text"]

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def normalize(series):
    return series.str.lower()

def remove_punctuation_numbers(series):
    return series.apply(lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x))

def remove_stopwords(series):
    stop = set(stopwords.words('english'))
    return series.apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

def tokenize(series):
    return series.apply(word_tokenize)

def stem(series):
    stemmer = PorterStemmer()
    return series.apply(lambda tokens: [stemmer.stem(word) for word in tokens])

def lemmatize(series):
    lemmatizer = WordNetLemmatizer()
    return series.apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])

def preprocess(series, use_stem=False, use_lemma=True):
    series = normalize(series)
    series = remove_punctuation_numbers(series)
    series = remove_stopwords(series)
    tokens = tokenize(series)

    if use_lemma:
        tokens = lemmatize(tokens)
    if use_stem:
        tokens = stem(tokens)

    processed_series = tokens.apply(lambda token_list: ' '.join(token_list))
    return processed_series, tokens.tolist()


preprocessed, docs = preprocess(text)


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.start_time = time.time()
    
    def on_epoch_begin(self, model):
        print(f"Epoch {self.epoch} started")
        self.start_time = time.time()
    
    def on_epoch_end(self, model):
        elapsed_time = time.time() - self.start_time
        print(f"Epoch {self.epoch} completed in {elapsed_time:.2f} seconds")
        self.epoch += 1

# Create the callback
epoch_logger = EpochLogger()

fasttext_model = FastText(
    sentences=docs,
    vector_size=300,
    window=5,
    min_count=2,
    workers=4,
    sg=1,
    callbacks=[epoch_logger],
    epochs=10
)

fasttext_model.callbacks = []
fasttext_model.save("fasttext.model")
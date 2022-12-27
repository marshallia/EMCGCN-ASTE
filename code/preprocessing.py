import json
import math
import pickle

import numpy as np
import spacy
import pandas as pd


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    data = pd.read_csv('../new_data/Yelp_2000_labeled.csv', index_col=0)
    data['tokens']=''
    data['pos']=''
    data['postag']=''
    data['deprel']=''
    data['head']=''
    data
    dependency = []

    docs = data['text'].apply(nlp)
    data['tokens'] = [[token.text for token in doc ] for doc in docs]
    data['deprel'] = [[token.dep_ for token in doc ]for doc in docs]
    data['pos'] =[ [token.pos_ for token in doc ]for doc in docs]
    data['postag'] = [[token.tag_ for token in doc ]for doc in docs]
    data['head'] = [[token.head.i for token in doc ]for doc in docs]
    data['sentiment'] = data['stars'].map({5: 'POS', 4: 'POS', 3: 'NEU', 2: 'NEG', 1: 'NEG'})
    data['sentence'] = data['text']
    data = data.drop(['text'], axis=1)

    adjective = [[token.text +'\\\B' if token.pos_ == 'ADJ' and token.dep_ == 'acomp' else token.text +'\\\O' for token in sent ] for sent in docs]  # adjective as description / opinion
    target = [[token  if token.dep_ == 'nsubj' and token.pos_ == 'NOUN' else token +'\\\O' for token.text in sent] for sent in docs]
    data
    results =[]
    for i in range(len(data)):
        results.append(dict(data.iloc[i]))
    json_object = json.dumps(results)

    with open("../new_data/yelp_train.json", "w") as outfile:
        outfile.write(json_object)




'''
Created on Jan 24, 2016

@author: Babis Kaidos
'''

import tarfile
import numpy
import os.path
import urllib.request
import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

def read_data():
    if not os.path.isdir('data/'):
        os.makedirs('data/')
        
    if not os.path.isdir('data/txt_sentoken'):
        if not os.path.isfile('data/review_polarity.tar.gz'):
            urllib.request.urlretrieve('http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz', filename='data/review_polarity.tar.gz')
        file = tarfile.open(name = 'data/review_polarity.tar.gz')
        file.extractall(path = 'data/')
    
    return sklearn.datasets.load_files('data/txt_sentoken', random_state=0)

def do_NB(data):
    reviews_classifier_NB = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB()),])
    parameters_NB = {'tfidf__ngram_range': [(1, 1), (1,2), (1, 3)], 'clf__alpha': numpy.arange(0.,0.2,0.005)}
    
    gs_NB = GridSearchCV(reviews_classifier_NB, parameters_NB, n_jobs=-1)
    gs_NB = gs_NB.fit(data.data, data.target)
    
    best_parameters, score, _ = max(gs_NB.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters_NB.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    print(score)
    
def do_linearSVC(data):
    reviews_classifier_lSVC = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC()),])
    parameters_lSVC = {'tfidf__ngram_range': [(1, 1), (1,2)],
                       'clf__tol': (0.0001, 0.0002, 0.0003),
                       'clf__C': (0.8, 0.9, 1.0, 1.1, 1.2),
                       'clf__loss': ('hinge', 'squared_hinge'),
                       'clf__max_iter': (100, 1000, 10000),
                       'clf__penalty': ('l1', 'l2'),
                       'clf__multi_class': ('ovr', 'crammer_singer')}
    
    gs_lSVC = GridSearchCV(reviews_classifier_lSVC, parameters_lSVC, n_jobs=-1)
    gs_lSVC = gs_lSVC.fit(data.data, data.target)
    
    best_parameters, score, _ = max(gs_lSVC.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters_lSVC.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    print(score)
    
if __name__ == '__main__':
    reviews_data = read_data()
    #do_nb(reviews_data)
    do_linearSVC(reviews_data)
'''
Created on Jan 24, 2016

@author: Babis Kaidos
'''

import tarfile
import time;
import numpy
import os.path
import urllib.request
import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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
    reviews_classifier_NB = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
    parameters_NB = {'tfidf__ngram_range': [(1, 1), (1, 2)], 'clf__alpha': numpy.arange(0.1, 2., 0.1)}
    
    gs_NB = GridSearchCV(reviews_classifier_NB, parameters_NB, n_jobs=-1, cv=10)
    gs_NB = gs_NB.fit(data.data, data.target)
    
    for parameters, mean_score, scores in gs_NB.grid_scores_:
        print(parameters)
        print(mean_score)
        print(scores)
        print('=======================')

    result_file = open('result.txt', 'a')
    print("Best score: %f" % gs_NB.best_score_, file = result_file)
    print("Best parameters: %r" % gs_NB.best_params_, file = result_file)
    localtime = time.asctime( time.localtime(time.time()) )
    print(localtime, file = result_file)
    result_file.close()
    
    return (gs_NB.best_score_, gs_NB.best_params_)

    
def do_linearSVC(data):
    reviews_classifier_lSVC = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])
    parameters_lSVC = {'tfidf__ngram_range': [(1, 1), (1,2)],
                       'clf__tol': (0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001),
                       'clf__C': (0.8, 0.9, 1.0, 1.1, 1.2),
                       'clf__loss': ('hinge', 'squared_hinge'),
                       'clf__max_iter': (100, 1000, 10000),
                       'clf__penalty': ('l2',),
                       'clf__multi_class': ('ovr', 'crammer_singer')}
    
    gs_lSVC = GridSearchCV(reviews_classifier_lSVC, parameters_lSVC, n_jobs=-1, cv=10)
    gs_lSVC = gs_lSVC.fit(data.data, data.target)
    
    for parameters, mean_score, scores in gs_lSVC.grid_scores_:
        print(parameters)
        print(mean_score)
        print(scores)
        print('=======================')
    
    result_file = open('result.txt', 'a')
    print("Best score: %f" % gs_lSVC.best_score_, file = result_file)
    print("Best parameters: %r" % gs_lSVC.best_params_, file = result_file)
    localtime = time.asctime( time.localtime(time.time()) )
    print(localtime, file = result_file)
    result_file.close()
    
    return (gs_lSVC.best_score_, gs_lSVC.best_params_)

def do_SVC(data):
    reviews_classifier_SVC = Pipeline([('tfidf', TfidfVectorizer()), ('clf', SVC())])
    parameters_SVC = {'tfidf__ngram_range': [(1, 1), (1,2)],
                      'clf__C': (0.8, 0.9, 1.0, 1.1, 1.2),
                      'clf__kernel': ('poly', 'rbf', 'sigmoid', 'precomputed'),
                      'clf__tol': (0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001),
                      'clf__decision_function_shape': ('ovr', 'ovo')}
    
    gs_SVC = GridSearchCV(reviews_classifier_SVC, parameters_SVC, n_jobs=-1, cv=10)
    gs_SVC = gs_SVC.fit(data.data, data.target)
    
    for parameters, mean_score, scores in gs_SVC.grid_scores_:
        print(parameters)
        print(mean_score)
        print(scores)
        print('=======================')

    result_file = open('result.txt', 'a')
    print("Best score: %f" % gs_SVC.best_score_, file = result_file)
    print("Best parameters: %r" % gs_SVC.best_params_, file = result_file)
    localtime = time.asctime( time.localtime(time.time()) )
    print(localtime, file = result_file)
    result_file.close()
    
    return (gs_SVC.best_score_, gs_SVC.best_params_)

def do_Decision_Tree(data):
    reviews_classifier_Decision_Tree = Pipeline([('tfidf', TfidfVectorizer()), ('clf', DecisionTreeClassifier())])
    parameters_Decision_Tree = {'tfidf__ngram_range': [(1, 1), (1, 2)],
                                'clf__criterion': ('gini', 'entropy'),
                                'clf__splitter': ('best', 'random')}
    
    gs_Decision_Tree = GridSearchCV(reviews_classifier_Decision_Tree, parameters_Decision_Tree, n_jobs=-1, cv=10)
    gs_Decision_Tree = gs_Decision_Tree.fit(data.data, data.target)
    
    for parameters, mean_score, scores in gs_Decision_Tree.grid_scores_:
        print(parameters)
        print(mean_score)
        print(scores)
        print('=======================')

    result_file = open('result.txt', 'a')
    print("Best score: %f" % gs_Decision_Tree.best_score_, file = result_file)
    print("Best parameters: %r" % gs_Decision_Tree.best_params_, file = result_file)
    localtime = time.asctime( time.localtime(time.time()) )
    print(localtime, file = result_file)
    result_file.close()
    
    return (gs_Decision_Tree.best_score_, gs_Decision_Tree.best_params_)
    
if __name__ == '__main__':
    reviews_data = read_data()
    #do_NB(reviews_data)
    do_linearSVC(reviews_data)
    do_SVC(reviews_data)
    do_Decision_Tree(reviews_data)

'''
Created on Jan 24, 2016

@author: Babis Kaidos
'''

import tarfile
import time
import numpy
import os.path
import urllib.request
import sklearn.datasets
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

def read_data():
    if not os.path.isdir('data/'):
        os.makedirs('data/')
        
    if not os.path.isdir('data/txt_sentoken'):
        if not os.path.isfile('data/review_polarity.tar.gz'):
            urllib.request.urlretrieve('http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz', filename='data/review_polarity.tar.gz')
        file = tarfile.open(name='data/review_polarity.tar.gz')
        file.extractall(path='data/')
    
    return sklearn.datasets.load_files('data/txt_sentoken', random_state=0)

def do_multinomialNB(data, target):
    reviews_classifier_NB = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
    parameters_NB = {'tfidf__ngram_range': [(1, 1), (1, 2)], 'clf__alpha': numpy.arange(0.1, 2., 0.1)}
    
    gs_NB = GridSearchCV(reviews_classifier_NB, parameters_NB, n_jobs=-1, verbose=1)
    
    start_time = time.time()
    gs_NB = gs_NB.fit(data, target)
    elapsed_time = time.time() - start_time
    
    print("Multinomial Naive Bayes results:")
    print("Best score: %f" % gs_NB.best_score_)
    print("Best parameters: %r" % gs_NB.best_params_)
    print("Time required: %f seconds" % elapsed_time)
    
    target_predicted = gs_NB.predict(reviews_data_test)
    print(metrics.classification_report(target_test, target_predicted, target_names=reviews_data.target_names))
    print(metrics.confusion_matrix(target_test, target_predicted))

    result_file = open('result.txt', 'a')
    print("Multinomial Naive Bayes results:", file=result_file)
    print("Best score: %f" % gs_NB.best_score_, file=result_file)
    print("Best parameters: %r" % gs_NB.best_params_, file=result_file)
    print(elapsed_time, file=result_file)
    print(metrics.classification_report(target_test, target_predicted, target_names=reviews_data.target_names), file=result_file)
    print(metrics.confusion_matrix(target_test, target_predicted), file=result_file)
    print("=================================================", file=result_file)
    result_file.close()
    
    return (gs_NB.best_score_, gs_NB.best_params_)

def do_BernoulliNB(data, target):
    reviews_classifier_NB = Pipeline([('tfidf', TfidfVectorizer()), ('clf', BernoulliNB())])
    parameters_NB = {'tfidf__ngram_range': [(1, 1), (1, 2)],
                     'clf__alpha': (0.00001, 0.0001, 0.001, 0.01, 0.1, 1),
                     'clf__binarize': (0.0001, 0.001, 0.01, 0.1, 1.),
                     'clf__fit_prior': (True, False)}
    
    gs_NB = GridSearchCV(reviews_classifier_NB, parameters_NB, n_jobs=-1, verbose=1)
    gs_NB = gs_NB.fit(data, target)
    
    for parameters, mean_score, scores in gs_NB.grid_scores_:
        print(parameters)
        print(mean_score)
        print(scores)
        print('=======================')

    result_file = open('result.txt', 'a')
    print("Best score: %f" % gs_NB.best_score_, file=result_file)
    print("Best parameters: %r" % gs_NB.best_params_, file=result_file)
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime, file=result_file)
    result_file.close()
    
    return (gs_NB.best_score_, gs_NB.best_params_)

def do_PassiveAggressive(data, target):
    reviews_classifier_PassiveAggressive = Pipeline([('tfidf', TfidfVectorizer()), ('clf', PassiveAggressiveClassifier())])
    parameters_PassiveAggressive = {'tfidf__ngram_range': [(1, 1), (1, 2)],
                                    'clf__C': (0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.),
                                    'clf__fit_intercept': (True, False),
                                    'clf__n_iter': (1, 2, 3, 5, 8, 13, 21),
                                    'clf__shuffle': (True, False),
                                    'clf__loss': ('hinge', 'squared_hinge'),
                                    'clf__warm_start': (True, False)}
    
    gs_PassiveAggressive = GridSearchCV(reviews_classifier_PassiveAggressive, parameters_PassiveAggressive, n_jobs=-1, verbose=1)
    gs_PassiveAggressive = gs_PassiveAggressive.fit(data, target)
    
    for parameters, mean_score, scores in gs_PassiveAggressive.grid_scores_:
        print(parameters)
        print(mean_score)
        print(scores)
        print('=======================')

    result_file = open('result.txt', 'a')
    print("Best score: %f" % gs_PassiveAggressive.best_score_, file=result_file)
    print("Best parameters: %r" % gs_PassiveAggressive.best_params_, file=result_file)
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime, file=result_file)
    result_file.close()
    
    return (gs_PassiveAggressive.best_score_, gs_PassiveAggressive.best_params_)

def do_Ridge(data, target):
    reviews_classifier_Ridge = Pipeline([('tfidf', TfidfVectorizer()), ('clf', RidgeClassifier())])
    parameters_Ridge = {'tfidf__ngram_range': [(1, 1), (1, 2)],
                        'clf__alpha': (0.001, 0.01, 0.1, 1., 10.),
                        'clf__fit_intercept': (True, False),
                        'clf__normalize': (True, False),
                        'clf__tol': (0.0001, 0.001,)}
    
    gs_Ridge = GridSearchCV(reviews_classifier_Ridge, parameters_Ridge, n_jobs=-1, verbose=1)
    gs_Ridge = gs_Ridge.fit(data, target)
    
    for parameters, mean_score, scores in gs_Ridge.grid_scores_:
        print(parameters)
        print(mean_score)
        print(scores)
        print('=======================')

    result_file = open('result.txt', 'a')
    print("Best score: %f" % gs_Ridge.best_score_, file=result_file)
    print("Best parameters: %r" % gs_Ridge.best_params_, file=result_file)
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime, file=result_file)
    result_file.close()
    
    return (gs_Ridge.best_score_, gs_Ridge.best_params_)

def do_SGD(data, target):
    reviews_classifier_SGD = Pipeline([('tfidf', TfidfVectorizer()), ('clf', SGDClassifier())])
    parameters_SGD = {'tfidf__ngram_range': [(1, 1), (1, 2)],
                        'clf__loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),
                        'clf__penalty': ('none', 'l2', 'l1', 'elasticnet'),
                        'clf__alpha': (0.000001, 0.00001, 0.0001, 0.001, 0.01),
                        'clf__l1_ratio': (0.15,),
                        'clf__fit_intercept': (True, False),
                        'clf__n_iter': (1, 2, 3, 5, 8, 13),
                        'clf__shuffle': (True, False),
                        'clf__warm_start': (True, False)}
    
    gs_SGD = GridSearchCV(reviews_classifier_SGD, parameters_SGD, n_jobs=-1, verbose=1)
    gs_SGD = gs_SGD.fit(data, target)
    
    for parameters, mean_score, scores in gs_SGD.grid_scores_:
        print(parameters)
        print(mean_score)
        print(scores)
        print('=======================')

    result_file = open('result.txt', 'a')
    print("Best score: %f" % gs_SGD.best_score_, file=result_file)
    print("Best parameters: %r" % gs_SGD.best_params_, file=result_file)
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime, file=result_file)
    result_file.close()
    
    return (gs_SGD.best_score_, gs_SGD.best_params_)

    
def do_linearSVC(data, target):
    reviews_classifier_lSVC = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])
    parameters_lSVC = {'tfidf__ngram_range': [(1, 1), (1, 2)],
                       'clf__tol': (0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001),
                       'clf__C': (0.8, 0.9, 1.0, 1.1, 1.2),
                       'clf__loss': ('hinge', 'squared_hinge'),
                       'clf__max_iter': (100, 1000, 10000),
                       'clf__penalty': ('l2',),
                       'clf__multi_class': ('ovr', 'crammer_singer')}
    
    gs_lSVC = GridSearchCV(reviews_classifier_lSVC, parameters_lSVC, n_jobs=-1, cv=10)
    gs_lSVC = gs_lSVC.fit(data, target)
    
    for parameters, mean_score, scores in gs_lSVC.grid_scores_:
        print(parameters)
        print(mean_score)
        print(scores)
        print('=======================')
    
    result_file = open('result.txt', 'a')
    print("Best score: %f" % gs_lSVC.best_score_, file=result_file)
    print("Best parameters: %r" % gs_lSVC.best_params_, file=result_file)
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime, file=result_file)
    result_file.close()
    
    return (gs_lSVC.best_score_, gs_lSVC.best_params_)

def do_SVC(data, target):
    reviews_classifier_SVC = Pipeline([('tfidf', TfidfVectorizer()), ('clf', SVC())])
    parameters_SVC = {'tfidf__ngram_range': [(1, 1), (1, 2)],
                      'clf__C': (0.8, 0.9, 1.0, 1.1, 1.2),
                      'clf__kernel': ('poly', 'rbf', 'sigmoid'),
                      'clf__tol': (0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001),
                      'clf__decision_function_shape': ('ovr', 'ovo')}
    
    gs_SVC = GridSearchCV(reviews_classifier_SVC, parameters_SVC, n_jobs=-1, verbose=1)
    gs_SVC = gs_SVC.fit(data, target)
    
    for parameters, mean_score, scores in gs_SVC.grid_scores_:
        print(parameters)
        print(mean_score)
        print(scores)
        print('=======================')

    result_file = open('result.txt', 'a')
    print("Best score: %f" % gs_SVC.best_score_, file=result_file)
    print("Best parameters: %r" % gs_SVC.best_params_, file=result_file)
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime, file=result_file)
    result_file.close()
    
    return (gs_SVC.best_score_, gs_SVC.best_params_)

def do_KNeighbors(data, target):
    reviews_classifier_KNeighbors = Pipeline([('tfidf', TfidfVectorizer()), ('clf', KNeighborsClassifier())])
    parameters_KNeighbors = {'tfidf__ngram_range': [(1, 1), (1, 2)],
                             'clf__n_neighbors': (13, 21, 34),
                             'clf__weights': ('uniform', 'distance'),
                             'clf__p': (1, 2)}
    
    gs_KNeighbors = GridSearchCV(reviews_classifier_KNeighbors, parameters_KNeighbors, n_jobs=-1, verbose=1)
    gs_KNeighbors = gs_KNeighbors.fit(data, target)
    
    for parameters, mean_score, scores in gs_KNeighbors.grid_scores_:
        print(parameters)
        print(mean_score)
        print(scores)
        print('=======================')

    result_file = open('result.txt', 'a')
    print("Best score: %f" % gs_KNeighbors.best_score_, file=result_file)
    print("Best parameters: %r" % gs_KNeighbors.best_params_, file=result_file)
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime, file=result_file)
    result_file.close()
    
    return (gs_KNeighbors.best_score_, gs_KNeighbors.best_params_)

def do_Decision_Tree(data, target):
    reviews_classifier_Decision_Tree = Pipeline([('tfidf', TfidfVectorizer()), ('clf', DecisionTreeClassifier())])
    parameters_Decision_Tree = {'tfidf__ngram_range': [(1, 1), (1, 2)],
                                'clf__criterion': ('gini', 'entropy'),
                                'clf__splitter': ('best', 'random'),
                                'clf__max_features': ('None', 'sqrt', 'log2')}
    
    gs_Decision_Tree = GridSearchCV(reviews_classifier_Decision_Tree, parameters_Decision_Tree, n_jobs=-1, cv=10)
    gs_Decision_Tree = gs_Decision_Tree.fit(data, target)
    
    for parameters, mean_score, scores in gs_Decision_Tree.grid_scores_:
        print(parameters)
        print(mean_score)
        print(scores)
        print('=======================')

    result_file = open('result.txt', 'a')
    print("Best score: %f" % gs_Decision_Tree.best_score_, file=result_file)
    print("Best parameters: %r" % gs_Decision_Tree.best_params_, file=result_file)
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime, file=result_file)
    result_file.close()
    
    return (gs_Decision_Tree.best_score_, gs_Decision_Tree.best_params_)

def do_Extra_Tree(data, target):
    reviews_classifier_Extra_Tree = Pipeline([('tfidf', TfidfVectorizer()), ('clf', ExtraTreeClassifier())])
    parameters_Extra_Tree = {'tfidf__ngram_range': [(1, 1), (1, 2)]}
    
    gs_Extra_Tree = GridSearchCV(reviews_classifier_Extra_Tree, parameters_Extra_Tree, n_jobs=-1, verbose=1)
    gs_Extra_Tree = gs_Extra_Tree.fit(data, target)
    
    for parameters, mean_score, scores in gs_Extra_Tree.grid_scores_:
        print(parameters)
        print(mean_score)
        print(scores)
        print('=======================')

    result_file = open('result.txt', 'a')
    print("Best score: %f" % gs_Extra_Tree.best_score_, file=result_file)
    print("Best parameters: %r" % gs_Extra_Tree.best_params_, file=result_file)
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime, file=result_file)
    result_file.close()
    
    return (gs_Extra_Tree.best_score_, gs_Extra_Tree.best_params_)

def do_Adaboost(data, target):
    reviews_classifier_AdaBoost = Pipeline([('tfidf', TfidfVectorizer()), ('clf', AdaBoostClassifier())])
    parameters_AdaBoost = {'tfidf__ngram_range': [(1, 1), (1, 2)],
                           'clf__base_estimator': (ExtraTreeClassifier(),),
                           'clf__algorithm': ('SAMME.R',)}
    
    gs_AdaBoost = GridSearchCV(reviews_classifier_AdaBoost, parameters_AdaBoost, n_jobs=-1, verbose=1)
    gs_AdaBoost = gs_AdaBoost.fit(data, target)
    
    for parameters, mean_score, scores in gs_AdaBoost.grid_scores_:
        print(parameters)
        print(mean_score)
        print(scores)
        print('=======================')

    result_file = open('result.txt', 'a')
    print("Best score: %f" % gs_AdaBoost.best_score_, file=result_file)
    print("Best parameters: %r" % gs_AdaBoost.best_params_, file=result_file)
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime, file=result_file)
    result_file.close()
    
    return (gs_AdaBoost.best_score_, gs_AdaBoost.best_params_)

def do_Extra_Trees(data, target):
    reviews_classifier_Extra_Trees = Pipeline([('tfidf', TfidfVectorizer()), ('clf', ExtraTreesClassifier())])
    parameters_Extra_Trees = {'tfidf__ngram_range': [(1, 1),],
                              'clf__n_estimators': (100000,)}
    
    gs_Extra_Trees = GridSearchCV(reviews_classifier_Extra_Trees, parameters_Extra_Trees, n_jobs=-1, verbose=2)
    gs_Extra_Trees = gs_Extra_Trees.fit(data, target)
    
    for parameters, mean_score, scores in gs_Extra_Trees.grid_scores_:
        print(parameters)
        print(mean_score)
        print(scores)
        print('=======================')

    result_file = open('result.txt', 'a')
    print("Best score: %f" % gs_Extra_Trees.best_score_, file=result_file)
    print("Best parameters: %r" % gs_Extra_Trees.best_params_, file=result_file)
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime, file=result_file)
    result_file.close()
    
    return (gs_Extra_Trees.best_score_, gs_Extra_Trees.best_params_)
    
if __name__ == '__main__':
    reviews_data = read_data()
    reviews_data_train, reviews_data_test, target_train, target_test = train_test_split(reviews_data.data, reviews_data.target, test_size=0.20, random_state=None)
    
    do_multinomialNB(reviews_data_train, target_train)
    do_BernoulliNB(reviews_data_train, target_train)
    do_linearSVC(reviews_data_train, target_train)
    do_SVC(reviews_data_train, target_train)
    do_Decision_Tree(reviews_data_train, target_train)
    do_Extra_Tree(reviews_data_train, target_train)
    do_Adaboost(reviews_data_train, target_train)
    do_Extra_Trees(reviews_data_train, target_train)
    do_PassiveAggressive(reviews_data_train, target_train)
    do_Ridge(reviews_data_train, target_train)
    do_SGD(reviews_data_train, target_train)
    do_KNeighbors(reviews_data_train, target_train)

import os
import re
import logging
import argparse
import pickle

import nltk
import nltk.tokenize
import nltk.stem

import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

nltk.download(['punkt', 'wordnet', 'stopwords'])

class DummyEstimator(BaseEstimator):
    """
    This is a Dummy estimator we want to use as placeholder since we want to search among different estimators
    to find the best one
    """
    def fit(self): pass

    def score(self): pass


def tokenize(text):
    """
    This function is used to replace a text with tokens. Before that urls will be replaced by an url pattern
    :param text: The rext we want to tokenize
    :return: Tokenized text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = nltk.tokenize.word_tokenize(text)
    lemmatizer = nltk.stem.WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class Modeller:
    """
    This is the class that encapsulates the logic for training our model based on the provided messages in a database
    It has a couple of properties that are required to run and only one central "run" method that needs to be called
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialization of our class with all required parameters
        :param args: The arguments passed to the application
        :raise: FileNotFoundError
        """
        self.database_file = args.database
        self.database_messages_table = args.messages_table
        self.database_categories_table = args.categories_table
        self.message_limit = args.message_limit
        self.model_file = args.model

        self.messages = None
        self.categories = None
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        self._check_if_files_exists()

    def run(self):
        """
        This is the central function that handles all the training and exporting of the model
        :raises: RuntimeError
        :return:
        """
        logging.info('Starting processing of data')

        self._load_data()
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2)

        self._build_model()

        logging.info('Training model...')
        self.model.fit(self.X_train, self.Y_train)

        self._evaluate_model()

        self._save_model()
        logging.info('Trained Model successfully exported')

    def _check_if_files_exists(self):
        """
        Check if a file exists on the hard disk. Raises an exception if the file does not exist
        :return: None
        """
        logging.debug('Checking if database file exists: %s', self.database_file)
        if not os.path.exists(self.database_file):
            logging.debug('File does not exist: %s', self.database_file)
            raise FileNotFoundError('Provided file does not exist: {}'.format(self.database_file))

    def _load_data(self):
        """
        This method is used to load the messages from the provided database file
        :raises:RunTimeError
        :return: None
        """
        logging.info('Loading messages from database')
        logging.info('Target DB %s', self.database_file)
        logging.debug('Creating database engine')
        engine = create_engine('sqlite:///{}'.format(self.database_file))
        logging.info('Loading data')
        logging.debug('Loading messages from database')

        query = None
        if self.message_limit < 0:
            query = '''
                SELECT
                    *
                FROM 
                    {TABLE}
            '''
        else:
            query = '''
                SELECT
                    *
                FROM 
                    {TABLE}
                limit {MESSAGELIMIT}
            '''
        messages_query = query.format(TABLE=self.database_messages_table, MESSAGELIMIT=self.message_limit)
        self.messages = pd.read_sql_query(messages_query, engine)

        if self.messages is None:
            self.logger.error('No messages found in database')
            raise RuntimeError('No messages found')

        logging.info('Loading categories from database')
        self.categories = pd.read_sql_table(self.database_categories_table, engine)

        if self.categories is None:
            self.logger.error('No categories found in database')
            raise RuntimeError('No categories found')

        logging.debug('Split X and Y data from messages')
        self.X = self.messages.message
        # self.Y = self.messages.medical_help
        self.Y = self.messages\
            .drop('message', axis=1)\
            .drop('id_messages', axis=1)\
            .drop('original', axis=1)\
            .drop('genre', axis=1)\
            .drop('id_categories', axis=1)\
            .drop('index', axis=1)

    def _build_model(self):
        """
        This method is responsible for building a Grid Search Model
        :return: None
        """
        logging.info('Building model...')
        pipeline = Pipeline(
            [
                (
                    'vect', CountVectorizer(tokenizer=tokenize,
                                            token_pattern=None)
                ),
                (
                    'tfidf', TfidfTransformer()
                ),
                (
                    'clf', DummyEstimator()
                )
            ]
        )

        parameters = [
            {
                'clf': [RandomForestClassifier()],
                'clf__bootstrap': [True],
                'clf__ccp_alpha': [0.0],
                'clf__class_weight': [None],
                'clf__criterion': ['gini'],
                'clf__max_depth': [None, 10, 50, 100],
                'clf__max_features':  ['sqrt'],
                'clf__min_impurity_decrease': [0.0],
                'clf__min_samples_leaf': [1],
                'clf__min_samples_split': [2],
                'clf__min_weight_fraction_leaf': [0.0],
                'clf__n_estimators': [50, 100, 200],
                'clf__oob_score': [False],
                'clf__random_state': [None],
                'clf__verbose': [0],
                'clf__warm_start': [False],
            },
            {
                'clf': [MLPClassifier()],
                'clf__max_iter': [500],
                'clf__hidden_layer_sizes': [(100,), (50, 50), (100, 100)],
            },
            # {
            #     'clf': [LinearSVC()]
            # },
            # {
            #     'clf': [DecisionTreeClassifier()]
            # },
            # {
            #     'clf': [KNeighborsClassifier(3)]
            # },
            # {
            #     'clf': [SVC(kernel="linear", C=0.025)]
            # },
            # {
            #     'clf': [GaussianProcessClassifier(1.0 * RBF(1.0))]
            # },
            # {
            #     'clf': [DecisionTreeClassifier(max_depth=5)]
            # },
            # {
            #     'clf': [AdaBoostClassifier()]
            # },
            # {
            #     'clf': [GaussianNB()]
            # },
            # {
            #     'clf': [QuadraticDiscriminantAnalysis()]
            # }
        ]

        self.model = GridSearchCV(pipeline,
                                  param_grid=parameters,
                                  n_jobs=-1)

    def _evaluate_model(self):
        """
        Evaluate model
        :return:
        """
        logging.info('Evaluate model...')
        y_pred = self.model.best_estimator_.predict(self.X_test)

        accuracy = (y_pred == self.Y_test).mean()

        logging.info('Evaluation results:')
        logging.info('Selected Classifier: %s', str(self.model.best_estimator_.named_steps['clf']))
        for index, value in accuracy.items():
            logging.info('Accuracy %s: %s', index, str(value))
        logging.info("Best Parameters: ")
        for key, value in self.model.best_params_.items():
            logging.info("%s: %s", key, value)

        logging.info('Classification report')
        report = classification_report(self.Y_test, y_pred, target_names=list(accuracy.index))
        for row in report.split('\n'):
            logging.info(row)

    def _save_model(self):
        """
        This method is used to save the best model to a file so that it can be used later in our Web App
        :return: None
        """
        best_model = self.model.best_estimator_
        logging.info('Writing model to file: %s', self.model_file)
        with open(self.model_file, 'wb') as fid:
            pickle.dump(best_model, fid)
        logging.debug('Model saved to file')


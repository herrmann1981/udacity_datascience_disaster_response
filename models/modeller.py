import os
import re
import logging
import argparse
import pickle

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
import nltk.tokenize
import nltk.stem

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import ClassifierChain


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

    @classmethod
    def tokenize(cls, text):
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

    def _build_model(self):
        """
        This method is responsible for building a Grid Search Model
        :return: None
        """
        logging.info('Building model...')
        pipeline = Pipeline(
            [
                (
                    'vect', CountVectorizer(tokenizer=Modeller.tokenize,
                                            token_pattern=None)
                ),
                (
                    'tfidf', TfidfTransformer()
                ),
                (
                    'clf', RandomForestClassifier()
                )
            ]
        )

        parameters = {
            'vect__analyzer': ['word'],
            'vect__binary': [False],
            'vect__decode_error':  ['strict'],
            'vect__encoding': ['utf-8'],
            'vect__input': ['content'],
            'vect__lowercase': [True],
            'vect__max_df': [1.0],
            'vect__max_features': [None],
            'vect__min_df': [1],
            'vect__ngram_range': [(1, 1)],
            'vect__stop_words': [None],
            'vect__strip_accents': [None],
            'vect__token_pattern': [None],
            'vect__vocabulary': [None],
            'tfidf__norm':  ['l2'],
            'tfidf__smooth_idf': [True],
            'tfidf__sublinear_tf': [False],
            'tfidf__use_idf': [True],
            'clf__bootstrap': [True],
            'clf__ccp_alpha': [0.0],
            'clf__class_weight': [None],
            'clf__criterion': ['gini'],
            'clf__max_depth': [None],
            'clf__max_features':  ['sqrt'],
            'clf__max_leaf_nodes':  [None],
            'clf__max_samples': [None],
            'clf__min_impurity_decrease': [0.0],
            'clf__min_samples_leaf': [1],
            'clf__min_samples_split': [2],
            'clf__min_weight_fraction_leaf': [0.0],
            'clf__n_estimators': [50],
            'clf__n_jobs': [None],
            'clf__oob_score': [False],
            'clf__random_state': [None],
            'clf__verbose' : [0],
            'clf__warm_start': [False],
        }

        self.model = GridSearchCV(pipeline,
                                  param_grid=parameters,
                                  scoring='f1_macro',
                                  n_jobs=-1)

    def _evaluate_model(self):
        """
        Evaluate model
        :return:
        """
        logging.info('Evaluate model...')
        y_pred = self.model.best_estimator_.predict(self.X_test)

        labels = np.unique(y_pred)
        confusion_mat = confusion_matrix(self.Y_test, y_pred, labels=labels)
        accuracy = (y_pred == self.Y_test).mean()

        logging.info('Evaluation results:')
        logging.info("Labels: %s", labels)
        logging.info("Confusion Matrix: %s", confusion_mat)
        logging.info("Accuracy: %s", accuracy)
        logging.info("Best Parameters: ")
        for key, value in self.model.best_params_.items():
            logging.info("%s: %s", key, value)

    def _save_model(self):
        """
        This method is used to save the best model to a file so that it can be used later in our Web App
        :return: None
        """
        best_model = self.model.best_estimator_
        logging.info('Writing model to file: %s', self.model_file)
        with open(self.model_file, 'wb') as fid:
            pickle.dump(best_model, fid, protocol=pickle.HIGHEST_PROTOCOL)
        logging.debug('Model saved to file')


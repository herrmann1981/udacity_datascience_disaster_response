import os
import sys
import logging
import argparse
import pandas as pd
from sqlalchemy import create_engine


class Processor:
    """
    This class is responsible for holding all the logic for processing the input messages and then clean the data
    and export the data to the database
    """

    def __init__(self, args: argparse.Namespace):
        """
        Constructor for our processor. Here we can initialize our parameters and set up everything we need to get
        going
        """
        self.messages_file = args.messages
        self.categories_file = args.categories
        self.database_file = args.database
        self.drop_duplicates = args.drop_duplicates
        self.messages_table = args.messages_table
        self.categories_table = args.categories_table

        self.messages: pd.DataFrame = None
        self.categories: pd.Series = None

        self._check_if_files_exists()

    def run(self):
        """
        Central entry point for our processor. This is the main method that needs to be called
        raises: RuntimeError
        :return: None
        """
        logging.info('Starting processing of data')
        logging.info('Messages file: %s', self.messages_file)
        logging.info('Categories file: %s', self.categories_file)
        logging.info('Target DB %s', self.database_file)

        logging.info('Loading data...')
        self._load_data()

        logging.info('Cleaning data...')
        self._clean_data()

        logging.info('Exporting data to database')
        self._save_data()

        logging.info('Cleaned data successfully exported')

    def _check_if_files_exists(self):
        """
        Check if the provided files exist on the hard disk. Raises an exception if the file does not exist
        :raises: FileNotFoundError
        :return: None
        """

        logging.info('Checking if files exist')
        files = [
            self.categories_file,
            self.messages_file
        ]
        for file in files:
            logging.debug('Checking if file exists: %s', file)
            if not os.path.exists(file):
                logging.debug('File does not exist: %s', file)
                raise FileNotFoundError('Provided file does not exist: {}'.format(file))

    def _load_data(self):
        """
        This method is responsible for loading the messages and categories from the provided csv files
        :return: None
        """
        logging.debug('Loading messages file: %s', self.messages_file)
        self.messages = pd.read_csv(self.messages_file)

        logging.debug('Loading categories file: %s', self.categories_file)
        df_categories = pd.read_csv(self.categories_file)

        logging.info('Splitting categories and converting it into numerical values')
        df_categories_split = self._expand_and_replace_categories_column_names(df_categories)
        logging.debug('Joining input data')
        self.messages = self.messages.join(df_categories_split,
                                           lsuffix='_messages',
                                           rsuffix='_categories')

    def _expand_and_replace_categories_column_names(self, df_categories: pd.DataFrame) -> pd.DataFrame:
        """
        This method is used to extract the categories columns into individual columns and then rename the
        columns with a meaningful name
        :param df_categories: The original categories dataframe that was read in from the csv
        :return: An updated dataframe of the categories
        """
        df_categories_temp = df_categories['categories'].str.split(pat=';', expand=True)
        first_row = df_categories_temp.iloc[0]
        self.categories = first_row.apply(lambda x: x[:-2])
        df_categories_temp.columns = self.categories

        for column in df_categories_temp:
            # take last character (0 or 1) and put it into the column
            df_categories_temp[column] = df_categories_temp[column].astype(str).str[-1]
            # convert the 0 or 1 into an integer
            df_categories_temp[column] = df_categories_temp[column].astype(int)

        result = pd.concat([df_categories['id'], df_categories_temp], axis=1)
        return result

    def _clean_data(self):
        """
        This method is responsible for cleaning duplicated entries in the dataframe and returns the cleaned
        dataframe
        :return:
        """
        if self.drop_duplicates == 'messageonly':
            logging.debug('Dropping duplicates based on message only')
            return self.messages.drop_duplicates(subset='message')
        else:
            logging.debug('Dropping exact duplicates')
            return self.messages.drop_duplicates()

    def _save_data(self):
        """
        Save the cleaned data to the target database
        :return: None
        """
        logging.debug('Creating database engine')
        engine = create_engine('sqlite:///{}'.format(self.database_file))

        logging.info('Writing messages to database')
        self.messages.to_sql(self.messages_table,
                             engine,
                             if_exists='replace',
                             chunksize=1000)

        logging.info('Writing categories to database')
        self.categories.to_sql(self.categories_table,
                               engine,
                               if_exists='replace',
                               chunksize=1000)


        logging.debug('Writing to database finished')

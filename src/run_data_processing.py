import sys
import logging
import argparse

from data.processor import Processor


def setup_arguments() -> argparse.Namespace:
    """
    Configure the input arguments and parse them
    :return: Parsed known arguments
    """
    parser = argparse.ArgumentParser(
        prog='python3 run_data_processing.py',
        description='Processes the raw data from disaster responses and stores it in a database',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-m'
        '--messages',
        type=str,
        help='The filepath to the csv containing all messages',
        dest='messages',
        action='store',
        default='../data/disaster_messages.csv'
    )

    parser.add_argument(
        '-c',
        '--categories',
        type=str,
        help='The filepath to the csv containing all categories',
        dest='categories',
        action='store',
        default='../data/disaster_categories.csv'
    )

    parser.add_argument(
        '-d',
        '--database',
        type=str,
        help='The filepath to database where we want to store all data',
        dest='database',
        action='store',
        default='../data/disaster.db'
    )

    parser.add_argument(
        '--messages_table',
        type=str,
        help='The table name where we want to store the disaster messages to.',
        dest='messages_table',
        action='store',
        default='Messages'
    )

    parser.add_argument(
        '--categories_table',
        type=str,
        help='The table name where we want to store the categories to.',
        dest='categories_table',
        action='store',
        default='Categories'
    )

    parser.add_argument(
        '-l',
        '--loglevel',
        type=str,
        help='The loglevel we want to use for the execution',
        dest='loglevel',
        action='store',
        choices=['error', 'warn', 'info', 'debug'],
        default='debug'
    )

    parser.add_argument(
        '--drop_duplicates',
        type=str,
        help='The specifies if we want to drop only exact duplicates or those where only the message is a duplicate '
             'but not the categories',
        dest='drop_duplicates',
        action='store',
        choices=['exact', 'messageonly'],
        default='exact'
    )

    return parser.parse_known_args()[0]


def setup_logging(loglevel: str):
    """
    Basic logging configuration that shall be used for this script
    :return:
    """
    level = logging.INFO
    if loglevel == 'error':
        level = logging.ERROR
    elif loglevel == 'warn':
        level = logging.WARNING
    elif loglevel == 'debug':
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s - %(message)s'
    )


def main():
    """
    Main method that encapsulates the whole logic of the application
    :return: None
    """
    args = setup_arguments()
    setup_logging(args.loglevel)

    try:
        processor = Processor(args)
        processor.run()

    except Exception as error:
        logging.error('Error during processing of data: %s', str(error))
        sys.exit(1)


if __name__ == '__main__':
    main()

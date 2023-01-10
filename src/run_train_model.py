import sys
import argparse
import logging

from models.modeller import Modeller


def setup_arguments() -> argparse.Namespace:
    """
    Configure the input arguments and parse them
    :return: Parsed known arguments
    """
    parser = argparse.ArgumentParser(
        prog='python3 run_train_model.py',
        description='Modeling the disaster response message data based on the messages stored in a database',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        help='The table name where to find the messages in the database',
        dest='messages_table',
        action='store',
        default='Messages'
    )

    parser.add_argument(
        '--categories_table',
        type=str,
        help='The table name where to find the categories in the database',
        dest='categories_table',
        action='store',
        default='Categories'
    )

    parser.add_argument(
        '--message_limit',
        type=int,
        help='The number of messages that shall be read from the database for training. This is especially useful'
             'if you don\'t have enough RAM / SWAP space on your computer to limit the consumption. The default value'
             '(-1) or omitting this parameter disables this feature.',
        dest='message_limit',
        action='store',
        default='-1'
    )

    parser.add_argument(
        '-m',
        '--model',
        type=str,
        help='The filepath where to store the trained model',
        dest='model',
        action='store',
        default='./models/trained_model.pkl'
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
    Main entry point of the application
    :return:
    """
    args = setup_arguments()
    setup_logging(args.loglevel)

    try:
        modeller = Modeller(args)
        modeller.run()

    except Exception as error:
        logging.error('Error during training the model: %s', str(error))
        sys.exit(1)


if __name__ == '__main__':
    main()

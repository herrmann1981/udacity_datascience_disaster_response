import argparse


class ArgumentParser:
    """
    Utility class for setting up the argument parsing
    """

    def __init__(self):
        """
        Constructor for our custom arguments
        """

        self.parser = argparse.ArgumentParser(
            prog='python3 run_webapp.py',
            description='Starting the web application for the disaster message classification',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        self.parser.add_argument(
            '-m',
            '--model',
            type=str,
            help='The pretrained model for the classification',
            dest='model',
            action='store',
            default='./models/trained_model.pkl'
        )

        self.parser.add_argument(
            '-d',
            '--database',
            type=str,
            help='The filepath to database where we want to store all data',
            dest='database',
            action='store',
            default='../data/disaster.db'
        )

    def parse(self) -> argparse.Namespace:
        """
        Parse the provided arguments from the command line
        :return:
        """
        return self.parser.parse_args()
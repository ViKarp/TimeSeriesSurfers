import logging


class BaseLogger:
    def __init__(self, log_file):
        logging.basicConfig(filename=log_file, level=logging.INFO)

    def log(self, message):
        logging.info(message)
        print(message)

import logging
import sys

class logger(object):
    def __init__(self, name, save_path=None, whether_stream=True, whether_file=False):
        # Check
        assert whether_stream or whether_file, "This log is meaningless."
        assert not (whether_file and save_path is None), "Log path is required, if you want to save the log."

        # Set the log
        self.log = logging.getLogger(name)
        self.log.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

        if whether_stream:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(self.formatter)
            self.log.addHandler(stream_handler)

        if whether_file:
            file_handler = logging.FileHandler(save_path, mode='w')
            file_handler.setFormatter(self.formatter)
            self.log.addHandler(file_handler)

    def info(self, message):
        self.log.info(message)

    def setLevel(self, level):
        self.log.setLevel(level)
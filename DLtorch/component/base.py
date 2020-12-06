from DLtorch.utils import getLogger

class BaseComponent(object):
    def __init__(self):
        self._logger = None
    
    @property
    def logger(self):
        if self._logger is None:
            self._logger = getLogger(self.__class__.__name__)
        return self._logger

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_logger" in state:
            del state["_logger"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # set self._logger to None
        self._logger = None
class base_dataset(object):
    def __init__(self, datatype, datadir, whether_valid):
        self.datatype = datatype
        self.datadir = datadir
        self.whether_valid = whether_valid
        self.datasets = {}
        self.datalength = {}

    @property
    def get_datatype(self):
        return self.datatype

    @property
    def get_datalength(self):
        return self.datalength

    @ property
    def dataset(self):
        return self.datasets
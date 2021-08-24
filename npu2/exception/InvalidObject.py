class InvalidObject(Exception):
    def __init__(self, object):
        self.object = object
        super().__init__("Object type not supported:%s" % object)
import datetime


class Logger:
    output = None

    def __init__(self, _output=None):
        self.output = _output

    def log(self, to_print):
        to_print = "{}: {}".format(str(datetime.datetime.now()), to_print)
        if self.output is not None:
            print(to_print, file=open(self.output, 'w'))
        else:
            print(to_print)

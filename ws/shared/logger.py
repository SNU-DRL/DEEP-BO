from __future__ import print_function
import os
import traceback

def print_trace(enable=True):
    ConsoleLogger().set_trace(enable)

def set_log_level(log_level):
    ConsoleLogger().set_level(log_level)
    if log_level == 'debug':
        print_trace(True)

def debug(msg):
    ConsoleLogger().debug(msg)

def warn(msg):
    ConsoleLogger().warn(msg)

def error(msg):
    ConsoleLogger().error(msg)

def log(msg):
    ConsoleLogger().log(msg)


class Singleton(object):
    _instances = {}
    def __new__(class_, *args, **kwargs):
        if class_ not in class_._instances:
            class_._instances[class_] = super(Singleton, class_).__new__(class_, *args, **kwargs)
        return class_._instances[class_]


class ConsoleLogger(Singleton):

    def __init__(self):
        # initialize first time only 
        if hasattr(self, 'cur_level') is False:
            self.cur_level = 'warn'
            self.levels = ['debug', 'warn', 'error', 'log']
            self.trace = False

    def set_level(self, log_level):
        if log_level.lower() in self.levels:
            self.cur_level = log_level.lower()

    def set_trace(self, trace):
        self.trace = trace

    def debug(self, msg):
        if self.levels.index(self.cur_level) <= self.levels.index('debug'):
            print("[{}:D] {}".format(os.getpid(), msg))

    def warn(self, msg):
        if self.levels.index(self.cur_level) <= self.levels.index('warn'):
            print("[{}:W] {}".format(os.getpid(), msg))
            self.print_trace()

    def error(self, msg):
        if self.levels.index(self.cur_level) <= self.levels.index('error'):
            print("[{}:E] {}".format(os.getpid(), msg))
            self.print_trace()

    def print_trace(self):
        if self.trace:
            exception_log = traceback.format_exc()
            if exception_log != None:
                print(exception_log) 

    def log(self, msg):
        print("[{}:L] {}".format(os.getpid(), msg))




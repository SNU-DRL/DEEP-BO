from __future__ import print_function
import os
import time
import logging
import logging.handlers
import traceback

__LOGGER__ = None

def print_trace(enable=True):
    global __LOGGER__
    if __LOGGER__ == None:
        __LOGGER__ = ConsoleLogger() # default logger
    __LOGGER__.set_trace(enable)

def set_log_file(log_file):
    global __LOGGER__
    print("Progress can be monitored via {}".format(log_file))
    try:
        __LOGGER__ = FileLogger()
        __LOGGER__.set_log_file(log_file)
    except Exception as ex:
        print(traceback.format_exc())


def set_log_level(log_level):
    global __LOGGER__
    if __LOGGER__ == None:
        __LOGGER__ = ConsoleLogger() # default logger    
    __LOGGER__.set_level(log_level)
    if log_level == 'debug':
        print_trace(True)

def debug(msg):
    global __LOGGER__
    if __LOGGER__ == None:
        __LOGGER__ = ConsoleLogger() # default logger    
    __LOGGER__.debug(msg)

def warn(msg):
    global __LOGGER__
    if __LOGGER__ == None:
        __LOGGER__ = ConsoleLogger() # default logger    
    __LOGGER__.warn(msg)

def error(msg):
    global __LOGGER__
    if __LOGGER__ == None:
        __LOGGER__ = ConsoleLogger() # default logger    
    __LOGGER__.error(msg)

def log(msg):
    global __LOGGER__
    if __LOGGER__ == None:
        __LOGGER__ = ConsoleLogger() # default logger    
    __LOGGER__.log(msg)


class Singleton(object):
    _instances = {}
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instances[cls]


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


class FileLogger(Singleton):

    def __init__(self):

        if hasattr(self, 'cur_level') is False:
            self.cur_level = 'warn'
            self.levels = ['debug', 'warn', 'error', 'log']
            self.trace = False
            self.logger = None

    def set_log_file(self, file_path):
        self.logger = logging.getLogger('FileLogger')
        formatter = logging.Formatter('[%(asctime)s][%(process)d:%(levelname)s] %(message)s')
        f_handler = logging.FileHandler(file_path)
        f_handler.setFormatter(formatter)
        self.logger.addHandler(f_handler)
        self.logger.setLevel(level=logging.DEBUG)

    def set_level(self, log_level):
        if self.logger == None:
            self.set_log_file('default.log')

        if log_level.lower() in self.levels:
            self.cur_level = log_level.lower()
            if self.cur_level == 'debug':
                self.logger.setLevel(level=logging.DEBUG)
            elif self.cur_level == 'warn':
                self.logger.setLevel(level=logging.WARN)
            elif self.cur_level == 'error':
                self.logger.setLevel(level=logging.ERROR)
            elif self.cur_level == 'log':
                self.logger.setLevel(level=logging.INFO)
            else:
                print('Invalid log level: {}'.format(log_level))    

    def set_trace(self, trace):
        self.trace = trace

    def debug(self, msg):
        if self.logger == None:
            self.set_log_file('default.log')
        if self.levels.index(self.cur_level) <= self.levels.index('debug'):
            self.logger.debug(msg)

    def warn(self, msg):
        if self.logger == None:
            self.set_log_file('default.log')
        if self.levels.index(self.cur_level) <= self.levels.index('warn'):
            self.logger.warn(msg)

    def error(self, msg):
        print(msg) # print message in console
        if self.logger == None:
            self.set_log_file('default.log')        
        if self.levels.index(self.cur_level) <= self.levels.index('error'):        
            self.logger.error(msg)

    def print_trace(self):
        if self.logger == None:
            self.set_log_file('default.log')

        exception_log = traceback.format_exc()
        if exception_log != None:
            self.logger.debug(exception_log) 

    def log(self, msg):
        print(msg) # reveal message in console also 
        if self.logger == None:
            self.set_log_file('default.log')        
        self.logger.info(msg)



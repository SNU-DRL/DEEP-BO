import os
import threading
import time
import traceback
import copy

import numpy as np

from ws.shared.logger import *


class WorkerResource:
    def __init__(self):
        self.id = 'cpu0' # default computing resource id

    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id


class Worker(object):
    def __init__(self, id=None):
        self.thread = None
        
        self.busy = False
        self.stop_flag = False
        self.thread_cond = threading.Condition(threading.Lock())
        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())
        self.timer = None
        self.timeout = None
        self.type = 'prototype'
        self.config = {}
        
        if id != None:
            self.id = id
        else:
            self.id = 'worker_proto'

    def get_id(self):
        return self.id

    def get_cur_status(self):
        if self.busy:
            if self.paused:
                return 'pending'
            else:
                return 'processing'
        else:
            if self.paused:
                return 'error'
            else:
                return 'idle'

    def start(self):
        with self.thread_cond:
            while self.busy:
                self.thread_cond.wait()
            self.busy = True

        if not self.timeout is None and not self.timer is None:
            self.timer.cancel()
        
        self.stop_flag = False
        #self.id += str(threading.current_thread().ident)
        self.thread = threading.Thread(
            target=self.execute, name='worker {} thread'.format(self.id))
        self.thread.daemon = True
        self.thread.start()

        if not self.timeout is None:
            self.timer = threading.Timer(self.timeout, self.stop)
            self.timer.daemon = True
            self.timer.start()
        return True

    def pause(self):
        self.paused = True
        self.pause_cond.acquire()
        #debug("Pause requested.")

    def resume(self):
        self.paused = False
        self.pause_cond.notify()
        self.pause_cond.release()
        #debug("Resume requested.")

    def stop(self):
        if not self.thread is None:
            #debug("Stop requested.")
            self.stop_flag = True
            if self.paused == True:
                self.resume()
            self.thread.join()

    def execute(self):
        raise NotImplementedError('execute() should be overrided.')


import os
import sys

# For path arrangement (set the parent directory as the root folder)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from ws.shared.lookup import *

if __name__ == '__main__':
    l = load('CIFAR10-ResNet', data_folder='./lookup/')
    print(l.num_hyperparams)
    l.get_all_test_acc_per_epoch()
    l.get_all_sobol_vectors()
    print(list([int(t) for t in l.get_all_exec_times()]))
    

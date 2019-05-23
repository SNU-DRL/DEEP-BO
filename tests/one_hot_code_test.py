import os
import sys
import numpy as np
# For path arrangement (set the parent directory as the root folder)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from ws.hpo.space_mgr import create_surrogate_space
from ws.hpo.utils.one_hot_grid import create_one_hot_grid

def test_main(surrogate, index):
    samples = create_surrogate_space(surrogate)
    grid = create_one_hot_grid(samples)
    print(samples.get_hpv(index))
    print("vector size: {}\nencoded: {}".format(len(grid), grid[index]))


if __name__ == "__main__":
    test_main('CIFAR10-ResNet', 6024)


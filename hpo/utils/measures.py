from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

class Measure(object):
    
    def difference(self, a, b):
        raise NotImplementedError()


class RankIntersectionMeasure(Measure):
    
    def __init__(self, true_errors):
        self.true_errors = np.asarray(true_errors)
        self.rank_index = np.asarray(true_errors).argsort()[:][::1]

    def intersection(self, candidates, acq_funcs, k, j=None):
        if j is None:
            j = len(candidates)
        true_ranks = self.rank_index[k:j]
        est_ranks = np.asarray(candidates)[np.asarray(acq_funcs).argsort()[:][::-1]][k:j]

        if len(true_ranks) != len(est_ranks):
            raise ValueError('est value is invalid.')
        n = len(true_ranks)
        if n > 0:
        #print("est rank: {}".format(est_ranks))
            intersect = set(true_ranks) & set(est_ranks)
            rate = float(len(intersect)/ n)
        else:
            intersect = set([])
            rate = 0.0
        
        return intersect, rate

    def compare_all(self, candidates, acq_funcs, 
                    ranges=None, bound=100, bin_size=50):
        if ranges is None:
            ranges = []
            for i in range(0, bound):
                ri = []
                ri.append(i * bin_size)
                i += 1
                if i * bin_size >= len(candidates):
                    ri.append(None)
                else:
                    ri.append(i * bin_size)

                if not None in ri:
                    ranges.append(ri)

        rates = []
        for ri in ranges:
            # accumulated rank intersection                 
            #_, r = self.intersection(candidates, acq_funcs, 0, ri[1])
            # rank intersection

            _, r = self.intersection(candidates, acq_funcs, ri[0], ri[1])
            rates.append(r*100)
        
        return rates                        
            



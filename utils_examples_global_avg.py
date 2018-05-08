from __future__ import print_function

import os, sys, time
import numpy as np
from collections import deque

from utils import *


def build_unique_examples(examplesHistory):
    """ Make examples unique by averaging policy_vector (pi) and value (v) for the same positions """
     
    print("build_unique_examples() history.length=", len(examplesHistory))
    start = time.time()

    stateMap = {}
    step = 0    
    initialCount = 0
    
    # list of lists
    # inner list contains tuples (position, policy_vector, reward, strRepr)
    for examples in examplesHistory:
        step += 1
        initialCount += len(examples)
        print("step:", step, ", examples.length:", len(examples))
        
        validate_random_sample(examples)
        
        for (position, policy_vector, reward, s) in examples:
            
            #s = strRepr(position)
            #print(s)
            #print(long_cap, no_progress, king_moves, repetition)
            
            # this code collects all samples for the same position for averaging them later
            if not s in stateMap:
                stateMap[s] = deque()
            stateMap[s].append((position, policy_vector, reward))
                
    print("stateMap.size:", len(stateMap))
    
    newExamples = deque()
    
    for s in stateMap.keys():

        """ this code averages multiple samples per position """
        predictions = stateMap[s]
        size = len(predictions)
        
        if size>1:
            sum_v = 0.
            sum_pi = []
            
            for (position,pi,v) in predictions:
                sum_v += v
                if len(sum_pi) == 0:
                    sum_pi = np.array(pi, dtype=np.float64, copy=True)
                else:
                    sum_pi += pi
            
            sum_v /= size
            sum_pi /= size
            
            assert len(sum_pi)==len(predictions[0][1]), str(len(sum_pi))+"!="+str(len(predictions[0][1])) 
            
            position = predictions[0][0]
            newExamples.append((position, sum_pi, sum_v))
        else:
            # the only example for the given position
            example = predictions[0]
            newExamples.append(example)
        
    print("total number of examples changed from", initialCount, "to", len(newExamples))
    full_time = time.time()-start
    print("time:", full_time, "s")

    return newExamples

from __future__ import print_function

import os, sys, time
import numpy as np
from collections import deque

from utils import *


def build_unique_examples_of_iteration(examples):
    """ Make examples unique by grouping all examples for the same positions. 
        If there are several samples for a position then average policy_vector (pi) for all of them.
    """
    validate_random_sample(examples)
    stateMap = {}
    # collect ALL samples for the equal positions
    for (position, policy_vector, reward, s) in examples:
        if not s in stateMap:
            stateMap[s] = deque()
        stateMap[s].append((position, policy_vector, reward, s))
    # average policy_vector for every position with several samples
    newExamples = deque()
    
    for s in stateMap.keys():
        
        predictions = stateMap[s]
        size = len(predictions)
        
        if size>1:
            sum_v = 0.
            sum_pi = []
            
            for (position,pi,v,strRepr) in predictions:
                sum_v += v
                if len(sum_pi) == 0:
                    sum_pi = np.array(pi, dtype=np.float64, copy=True)
                else:
                    sum_pi += pi
            
            sum_v /= size
            sum_pi /= size
            
            assert len(sum_pi)==len(predictions[0][1]), str(len(sum_pi))+"!="+str(len(predictions[0][1])) 
            
            position = predictions[0][0]
            newExamples.append((position, sum_pi, sum_v, s))
        else:
            # the only example for the given position
            example = predictions[0]
            newExamples.append(example)
            
    print("build_unique_examples_of_iteration(): total number of examples changed from", len(examples), "to", len(newExamples))

    return newExamples

def build_unique_examples(examplesHistory):
    """ Make examples unique by overriding older examples by newer ones for the same positions.
        Examples of every iteration are processed by special algorithm, see build_unique_examples_of_iteration()
    """
     
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
        
        # samples of the step should be made unique (select by max reward) before global processing
        examples = build_unique_examples_of_iteration(examples)
        
        for (position, policy_vector, reward, s) in examples:
            
            #s = strRepr(position)
            #print(s)
            #print(long_cap, no_progress, king_moves, repetition)
            
            # this code collects all samples for the same position for averaging them later
            #if not s in stateMap:
            #    stateMap[s] = deque()
            #stateMap[s].append((position, policy_vector, reward))

            # this code holds the only sample per position, i.e. more recent sample overrides the previous one(s)
            # NB! examplesHistory should be collected starting from older iterations to recent ones
            stateMap[s] = (position, policy_vector, reward)
                
    print("stateMap.size:", len(stateMap))
    
    newExamples = deque()
    
    for s in stateMap.keys():
        
        """ this code copies the only sample per position to newExamples """
        newExamples.append(stateMap[s]) # tuple of (position, policy_vector, reward)
        
        """ this code averages multiple samples per position 
        
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
        """
        
    print("total number of examples changed from", initialCount, "to", len(newExamples))
    full_time = time.time()-start
    print("time:", full_time, "s")

    return newExamples

import pandas
import numpy as np
#cimport numpy as np
import schemas.image_list as i
import utils 
import math
import time
import cython
import imageio
import os
import ntpath
from numba import jit

@jit(cache=True, nopython=True)
def computeModelSplits(total_samples, group_size_to, group_size_from, len_df_category_list):
    set_sizes = []
    iteration = 1
    while total_samples > 0:
        total_samples -= int(group_size_to // iteration)
        if total_samples<group_size_from or total_samples < len_df_category_list:
            total_samples += int(group_size_to // iteration)
            set_sizes.append(int(total_samples))
            break

        else:
            set_sizes.append(group_size_to // iteration)
        
        if group_size_to // (iteration + 1) <= group_size_from:
            iteration = 1

        else:
            iteration += 1
        
    return set_sizes

@jit(cache=True, nopython=True)
def nextIndex(i, set_index, set_sizes, proportions, set_increments):
    while set_index < len(set_sizes) and np.floor(set_sizes[set_index]*proportions[i]) <= set_increments[set_index]:
        set_index += 1
    
    if set_index >= len(set_sizes):
        set_index = 0
        while set_index < len(set_sizes) and np.floor(set_sizes[set_index]*proportions[i]) <= set_increments[set_index]:
            set_index += 1

        if set_index >= len(set_sizes):
            set_index = 0
    
    return set_index
import sys
import units.unit as unit
#sys.path.insert(1, '../')
#from preprocessing.schemas.image_list import ImageList 
import pandas
import numpy as np
#cimport numpy as np
import schemas.image_list as i
import utils 
import math
import time
import cython
import imageio
from numba import jit
import numba

@jit(cache=True, nopython=False)
def filterImage(img, filter):
        print("shape ", img.shape)
        fft_img = utils.fourierTransform(img)
        fft_img*=filter
        img_reco = utils.inv_FFT_all_channel(fft_img)
        #return np.abs(img_reco.real)
        return np.abs(img_reco.real).astype(np.uint8), fft_img

@jit(cache=True)
def gaussianResponse(x, y, factorx, factory):
    return math.exp(-( x**2/(2*factorx)+y**2/(2*factory) ))#1/( 1 + x**2/(factorx**2)+y**2/(factory**2) )

def getSquareFilter(shape):
    square_filter = utils.draw_square(shape, 1)
    return square_filter

@jit(cache=True)
def contrastTunning(img, contrast_curve_step):
    min = np.min(img)
    max = np.max(img)
    t = img - min/2 
    t = t/(max-min/4) #the intervals

    out = (1-t)**3 * contrast_curve_step[0] + 3* t * (1-t)**2 * contrast_curve_step[1] + 3*(1-t)*t**2 * contrast_curve_step[2] + t**3 * contrast_curve_step[3]
    #out = (1-t)**3 * contrast_curve_values[0] + 3*(1-t)**2*t * contrast_curve_values[1] + 3*t**2*(1-t) * contrast_curve_values[2] + t**3 * contrast_curve_values[3]
    return out#.astype(np.uint8)


@jit(cache=True, nopython=True, error_model="numpy")
def equalizationFilter(fft, direction_pass, direction_filter,  ponderation_distance_factor, symmetry,  kernel_size, std_x, std_y, square_proportion, double_inverse):
            #A = np.zeros(shape,dtype=float).reshape(-1)
        #B = self.getSquareFilter(shape).reshape(-1)

        #cdef float coordx, coordy, distance, dot, pond,  gaussian, square, inverse, inverse2
        #cdef Py_ssize_t a, s, minx, maxx, miny, maxy, offset

        shape = fft.shape
        center = np.array((shape[0]/2.0, shape[1]/2.0))
        gaussian_factor_x = center[1] * std_x
        gaussian_factor_y = center[0] * std_y
        distance = np.linalg.norm(center) * ponderation_distance_factor

        #filter = np.zeros_like(fft, dtype=float)
        #filter_reshape = filter.reshape(-1)
        #for x in range(len(fft_loop)):
        #    a = x // shape[0]
        #    s = x % shape[1]
        
        offset = int((shape[0] * (1 - square_proportion)) * 0.5)

        #cdef float[:,:] direction_pass = direction_pass_in
        
        
        for a in range(shape[0]):
            for s in range(shape[1]):
                if 1 == 0:
                    print("the stupid hint for the compiler")

                coord_x = (a-center[0])*direction_pass[0] + (s-center[1]) * direction_pass[1]
                coord_y = (a-center[0])*direction_filter[0] + (s-center[1]) * direction_filter[1]
                gaussian = gaussianResponse(coord_x, coord_y, gaussian_factor_x * shape[1], gaussian_factor_y * shape[0])
                square = 1
                if a < offset:
                    square = 0
                if a > shape[0] - offset:
                    square = 0
                if s < offset:
                    square = 0
                if s > shape[1] - offset:
                    square = 0
                
                #for a in range(len(A)):
                #    for s in range(len(A[a])):
                #origin_distance = np.linalg.norm([a-center[0], s-center[1]])
                dot = (a-center[0])*direction_pass[0] + (s-center[1]) * direction_pass[1]
                if  symmetry:
                    dot=abs(dot)

                else:
                    if dot < 0:
                        dot=0

                pond = dot/distance
                #ramp = origin_distance/max_distance   
                if pond > 1:
                    pond = 1     


                #filter[a][s] = 2 * high_amp * A[a][s]*(1-pond**2) + B[a][s] * pond**2 * high_amp #* (2 - ramp) #* dot / max_distance
                #filter_value = 1 *A[a][s] *(1-pond**2) + B[a][s] * pond**2 *1 #* high_amp #* (2 - ramp) #* dot / max_distance
                filter_value = 1 * gaussian *(1-pond**2) + square * pond**2 *1 #* high_amp #* (2 - ramp) #* dot / max_distance
                #filter[a][s] = filter_value
                if filter_value < 1:
                    inverse2 = 0
                    inverse = 0
                    minx = s-kernel_size//2
                    maxx = s+kernel_size//2
                    miny = a-kernel_size//2
                    maxy = a+kernel_size//2
                    if miny < 0:
                        miny = 0
                    if maxy >=  fft.shape[0]:
                        maxy = fft.shape[0]-1
                    if minx < 0:
                        minx = 0
                    if maxx >=  fft.shape[1]:
                        maxx = fft.shape[1]-1    
                        
                    sl1 = np.abs(fft[miny:maxy,minx:maxx])
                    #sl = sl1.reshape(-1)
                    suma = 0
                    for yy in range(sl1.shape[0]):
                        for xx in range(sl1.shape[1]):
                            energy = sl1[yy][xx] #sl[i].real**2 + sl[i].imag**2
                           # inverse +=  1/energy
                            inverse2 += 1/ energy**2
                            if energy < 1:
                                inverse += energy *  1.0 / (1.0 + np.abs(energy))
                                suma += 1.0 / (1.0 + np.abs(energy))
                                
                            else:
                                suma += 1
                                inverse += 1.0 / energy
                    #for a1 in range(kernel_size):
                    #    p1 = a+a1-kernel_size//2
                    #    if p1 >=fft.shape[0]:
                    #        p1 -= fft.shape[0]
                        
                    #    for s1 in range(kernel_size):
                    #        p2 = s+s1-kernel_size//2
                    #        if p2 >=fft.shape[1]:
                    #            p2 -= fft.shape[1]

                    #        energy = np.abs(fft[p1][p2])
                    #        inverse +=  1
                    #        inverse2 += 1 / energy
                    if inverse != 0:
                        if double_inverse and inverse2 != 0 : 
                            inverse = inverse / inverse2
                        
                        else:
                            inverse = (sl1.shape[0] * sl1.shape[1]) / inverse
                    absolute = np.abs(fft[a][s])
                    unit_complex = fft[a][s]  / absolute
                    fft[a][s] = unit_complex * ( (1-filter_value) * inverse + absolute * (filter_value) )#* high_amp #* (2 - ramp) #* dot / max_distance
                    #fft_loop[x] = fft_loop[x] / np.abs(fft_loop[x]) * ( (1-filter_value) * inverse + np.abs(fft_loop[x]) * (filter_value) )#* high_amp #* (2 - ramp) #* dot / max_distance
        #fft /= absolutes
        return fft
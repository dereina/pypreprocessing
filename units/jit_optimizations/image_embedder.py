import numpy as np
import imageio
from PIL import Image
from skimage import transform, io
from itertools import product
import os, sys
import matplotlib.pyplot as plt
import math 
from scipy import ndimage, misc
from contextlib import contextmanager
import pickle
import time;
from scipy import stats
from pathlib import Path

from numba import njit, jit
import itertools
import utils
import pylab as pl

#static module functions
def test_stat(y):
    print(y.shape)
    return np.mean(y)

@jit(cache=True, nopython=True) 
def direct_stat(y):
    #np.array(list(map(f, x)))
    out = 0
    suma=0
    for i in range(len(y)):
        out += y[i] * np.abs(y[i]) *0.0000000001
        suma += np.abs(y[i]) *0.0000000001
    
    if suma !=0:
        out /=suma
    
    return out
@jit(cache=True, nopython=True) 
def direct_plus_stat(y):
    #np.array(list(map(f, x)))
    out = 0
    suma=0
    for i in range(len(y)):
        out += y[i] * (1+np.abs(y[i])) *0.0000000001
        suma += (1+np.abs(y[i])) *0.0000000001
    
    if suma !=0:
        out /=suma
    
    return out

@jit(cache=True, nopython=True) 
def inverse_plus_stat(y):
    out = 0
    suma=0
    for i in range(len(y)):
        out += y[i] * 1/(1 + np.abs(y[i]))
        suma += 1/(1 + np.abs(y[i]))
    
    if suma !=0:
        out /= suma
    
    return out

@jit(cache=True, nopython=True) 
def inverse_stat(y):
    out = 0
    suma=0
    for i in range(len(y)):
        if y[i] < 1:
            out += y[i] *  1.0 / (1.0 + np.abs(y[i]))
            suma += 1.0 / (1.0 + np.abs(y[i]))
            
        else:
            suma += 1
            out += 1.0 / y[i]

    if suma !=0:
        out = out/suma
    
    return out

@jit(cache=True, nopython=True) 
def mean_stat(y):
    out = 0
    suma=0
    #cells = [[x*y for y in range(5)] for x in range(10)]
    #for x,y in itertools.product(range(10), range(5)):
    #    print("(%d, %d) %d" % (x,y,cells[x][y]))
    for i in range(len(y)):
        suma += y[i]
    
    if suma !=0:
        out = suma / len(y)
    
    return out

@jit(cache=True, nopython=True) 
def test_stats(arr):
    d =direct_stat(arr)
    dp =direct_plus_stat(arr)
    ip =inverse_plus_stat(arr)
    i = inverse_stat(arr)
    m = mean_stat(arr)
    print("test_stats")
    print(arr)
    print(d)
    print(dp)
    print(i)
    print(ip)
    print(m)
    print(0.5*(d + i)) #below mean when more than two components, mean if not
    print(0.5*(dp + ip)) #below mean mean when more than two components, mean if not
    print(0.5*(dp + i)) #below mean
    print(0.5*(d + ip)) #above mean

@jit(cache=True) 
def powerHarmonicsEmbeddingForSingleImage(name, affination, plot_output):
    now = time.time()
    image, shape = utils.getImageFromFileMosaicToAffination(name, affination)# getImageFromFileGetShape(name, affination, affination, True)
    fourier_image = np.fft.fft2(image)
    height = shape[0]
    width = shape[1]

    #the_figures = [ InputPFF(image, "mosaic")]
    #plot_figures(the_figures)

    #print(parameters)
    """
    try:
        parameters = row['image'].split(".")[-2].split("_")[1:-1]
        exposure = float(parameters[0][3::])
        gain = float(parameters[1][1::])

    except:

        exposure = 21490
        gain = 500
    """
    #image = mpimg.imread("clouds.png")
    #print(image.shape)
    fourier_amplitudes = np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(image.shape[0]) * image.shape[1]# crea un vector de 0 a 499 seguido de -500 hasta -1, esto si el vector es de 1000 puntos y se multiplca cadaa muestra por 1000 para ponerlo entre [0, 1]
    #print("kfreq")
    #print(kfreq)
    kfreq2D = np.meshgrid(kfreq, kfreq) #crea un array con el kfreq puesto en fila y otro con el kfreq puesto en columnas, es el layout de de la transformada 2D
    #print("kfreq2D")
    #print(kfreq2D)
    #print("kfreq2D[0]")
    #print(kfreq2D[0])
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2) #resulta que son wave vectors, y aqui calculas su norma... - _-....
    #print("knrm")
    #print(knrm)
    knrm = knrm.flatten() #esto convierte a array 1D, concatena las filas
    #print("knrm flatten")
    #print(knrm)
    fourier_amplitudes = fourier_amplitudes.flatten() #lo mismo con las amplitudes de la TF

    #kbins = np.arange(0.5, image.shape[0]/2 +1, 1.)
    kbins = np.arange(0.5, affination//2+1., 1)

    kvals = 0.5 * (kbins[1:] + kbins[:-1] )#esto lo hace porque el arrange lo hace como lo hace, itera des de 1 hasta al final y promedia con lo mismo des de 0 hasta el penultimo incluido. punto a punto... y le queda 1 2 3.... 500 (los midpoints)
    
    if len(fourier_amplitudes) != len(knrm):
        print("why?!!!!!")
        print(len(fourier_amplitudes))
        print(len(knrm))

    #Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        #statistic = "mean",
                                        #statistic =lambda y: np.percentile(y, 10),
    #                                    statistic= Unit.mean_stat,
    #                                    bins = kbins)

    direct, inverse  =  utils.getImageLuminosities(image)

    Abinsd, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = direct_stat,
                                        #statistic =lambda y: np.percentile(y, 10),
                                        #statistic= Unit.direct_stat,
                                        bins = kbins)

    Abinsi, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        #statistic = "mean",
                                        #statistic =lambda y: np.percentile(y, 10),
                                        statistic= inverse_stat,
                                        bins = kbins)
    #Abinsiplus, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        #statistic = "mean",
                                        #statistic =lambda y: np.percentile(y, 10),
    #                                    statistic= Unit.inverse_plus_stat,
    #                                    bins = kbins)

    #parece que queremos la varianza total en cada bin, ahora tenemos el promedio del power, para obtener el power tebnemos que multiplicar por el volumne en cada bin
    Abinsd *= 4. * np.pi / 3. * (kbins[1:]**3 - kbins[:-1]**3)
    Abinsi *= 4. * np.pi / 3. * (kbins[1:]**3 - kbins[:-1]**3)
    #the_figures = [InputPFF(image, " normal")]
    #the_figures_fou = [InputPFF(fourier_image, " f amplitudes")]
    #plot_figures(the_figures, "l")
    #plot_fourier_figures(the_figures_fou)
    #max = np.max(Abins)
    #Abins /= max
    Abinsd /= np.max(Abinsd)
    Abinsi /= np.max(Abinsi)
    #a = np.concatenate((kvals,kvals + np.max(kvals)))
    #reversed = np.flipud(Abinsd)
    #pl.plot(np.concatenate((kvals, kvals + np.max(kvals))), 10**(np.concatenate((reversed,Abinsi))))
    #pl.show()
    #pl.plot(np.concatenate((kvals, kvals + np.max(kvals))), 2 - np.log10(np.concatenate((reversed,Abinsi))))
    #pl.show()
    #pl.plot(np.concatenate((kvals, kvals + np.max(kvals))), np.concatenate((Abinsi,Abins)))
    #pl.show()

    #pl.plot(np.concatenate((kvals, kvals + np.max(kvals))), np.concatenate((Abinsi,reversed)))
    #pl.show()


    #pmode, pmedian, pstd, pdirect, pinverse, pinterval, pdirect_sum =  utils.getVectorLuminosities(Abinsd)
    #pinterval[1] = max
    
    
    #neww = [height/width, direct, inverse]
    #neww += [height, width, mode, median, std, direct, inverse, interval[0], interval[1], pmode,  pmedian, pstd, pdirect, pinverse, pinterval[0], pinterval[1]]
    #pl.plot(np.concatenate((kvals, kvals + np.max(kvals))), np.array((2 - np.log10(np.concatenate((reversed,Abinsi)))).tolist()))
    #pl.show()
    neww = [height/width, direct, inverse] + (2 - np.log10(2.2250738585072014e-308 + np.concatenate((np.flipud(Abinsd),Abinsi)))).tolist()
    if plot_output:
        pl.plot(np.concatenate((kvals, kvals + np.max(kvals))), np.array((2 - np.log10(2.2250738585072014e-308 + np.concatenate((np.flipud(Abinsd),Abinsi)))).tolist()))
        pl.show()
    #neww += [Abinsi[i] for i in range(len(Abinsi))]
    #neww += [Abinsd[i] for i in range(len(Abinsd) - 1, -1, -1)]
    print("embedding Ellapsed ", time.time() - now)                                    

    return neww

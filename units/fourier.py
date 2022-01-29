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
#import psyco
#psyco.full()

class Unit(unit.Unit):
    def __init__ (self, context, unit_config= None, input_map_id_list = None): #the data schema of the input map is checked by this Unit
        unit.Unit.__init__(self, context, unit_config, input_map_id_list)
        print("__init__ fourier")
        #load data from sources, read configuration file
        self.context = context #application context, to get global affination, root, and global parameters in general...
        #an empty list means no inputs... the default value is provided...
        self.df_list = self.getInputOrDefault(type(pandas.DataFrame()), []) #inputs ar always lists, should be a struct with meta data from the sender...
        #if self.inputs is not None:
        #    print(type(pandas.DataFrame()))
        #    if type(pandas.DataFrame()) in self.inputs:
        #        self.df_list = self.inputs[type(pandas.DataFrame())]

        self.image = self.getConfigOrDefault('image', 'image')
        self.affination = self.getConfigOrDefault('affination', 1024)
        self.std_x = self.getConfigOrDefault('std_x', 1.0)
        self.std_y = self.getConfigOrDefault('std_y', 0.00005)
        self.ponderation_distance_factor = self.getConfigOrDefault('ponderation_distance_factor', 0.05)
        self.theta_pass = np.deg2rad(self.getConfigOrDefault('theta_pass', 0))
        self.symmetry = self.getConfigOrDefault('symmetry', True)
        self.kernel_size = self.getConfigOrDefault('kernel_size', 1)
        self.direction_pass = [np.sin(self.theta_pass), np.cos(self.theta_pass)] #[y, x]
        self.direction_filter = [-np.cos(self.theta_pass), np.sin(self.theta_pass)] #clock wise i think...
        self.square_proportion = self.getConfigOrDefault('square_proportion', 1)
        self.normalization_value = self.getConfigOrDefault('normalization_value', 0)
        self.contrast_curve_step = self.getConfigOrDefault('contrast_curve_step', [0, 0.3334, 0.6667, 1])
        #load datafrom inputs...
    def contrastTunning(self, img, contrast_curve_step):
        offset = img-np.min(img) 
        t = offset/np.max(offset) #the intervals

        out = (1-t)**3 * contrast_curve_step[0] + 3* t * (1-t)**2 * contrast_curve_step[1] + 3*(1-t)*t**2 * contrast_curve_step[2] + t**3 * contrast_curve_step[3]
        #out = (1-t)**3 * contrast_curve_values[0] + 3*(1-t)**2*t * contrast_curve_values[1] + 3*t**2*(1-t) * contrast_curve_values[2] + t**3 * contrast_curve_values[3]
        return out#.astype(np.uint8)

    def run(self):
        #get inputs from input_map_id_list, the lasts inserts is what you need, unless you want to use the whole list...
        print("run fourier ")
        new_width = self.affination
        new_height = self.affination
        if self.df_list is not None:
            for df in self.df_list:
                for index, row in df.iterrows():
                    start = time.time()
                    img = utils.getImageFromFile(row[self.image], new_width, new_height, True) #if new_width is 0 is not resizing...
                    fft_original = utils.fourierTransform(img)
                    #gaussian = self.getGaussianFilter(fft_original.shape)
                    #square = self.getSquareFilter(fft_original.shape)
                    fft, filter =  self.equalizationFilter(fft_original, self.direction_pass, self.direction_filter, self.ponderation_distance_factor, self.symmetry, self.kernel_size)
                    #fft = self.energyEqualization(fft_original, out, self.direction_pass, self.ponderation_distance_factor, self.symmetry, self.kernel_size)
                    end = time.time()
                    print("Ellapsed: ")
                    print(end - start)
                    #filter = self.filterPonderation(gaussian, square, self.direction_pass, self.ponderation_distance_factor, self.symmetry)
                    inv_filter = np.fft.ifftshift(np.abs(utils.inv_FFT_all_channel(filter)))
                    equalized = np.abs(utils.inv_FFT_all_channel(fft))
                    #equalized = (equalized/np.max(equalized)*255).astype(np.uint8)
                    contrasted=self.contrastTunning(equalized, self.contrast_curve_step)#.astype(np.uint8)
                    path = self.getPathOutputForFile(row[self.image])
                    imageio.imwrite(uri=path, im=equalized[0:500, 500:1024])  
                    imageio.imwrite(uri=path+"contrasted.png", im=contrasted)  
                    imageio.imwrite(uri=path+".png", im=equalized)  

                    img_filtered, fft_filtered = self.filterImage(img, filter)
                    fft_filtered2 = utils.fourierTransform(img_filtered)
                    
                    the_figures = [utils.InputPFF(utils.fft_img_values(np.abs(fft_original)), "original fft"), utils.InputPFF(inv_filter, "convolution filter"),  utils.InputPFF(filter, "filter")]
                    the_figures1 = [utils.InputPFF(utils.fft_img_values(np.abs(fft_filtered)), "filtered fft"), utils.InputPFF(img, row[self.image], 0, 0),  utils.InputPFF(img_filtered, "filtered", 0, 0)]
                    the_figures2 = [utils.InputPFF(utils.fft_img_values(np.abs(fft)), "equalized fft"), utils.InputPFF(img, row[self.image], 0, 0),  utils.InputPFF(equalized, "equalized", 0, 0)]
                    the_figures3 = [utils.InputPFF(utils.fft_img_values(np.abs(fft_filtered)), "filtered fft"), utils.InputPFF(utils.fft_img_values(np.abs(fft)), "equalized fft"),  utils.InputPFF(utils.fft_img_values(np.abs(fft_original)), "original fft")]

                    #imageio.imwrite(uri=self.context.origin+"/"+key+"-"+key2+"-sum.bmp", im=img0)  
                    #imageio.imwrite(uri=self.context.origin+"/"+key+"-"+key2+"-direct.bmp", im=img1)  
                    #imageio.imwrite(uri=self.context.origin+"/"+key+"-"+key2+"-inverse.bmp", im=img2)  

                    utils.plot_figures(the_figures)
                    utils.plot_figures(the_figures1)
                    utils.plot_figures(the_figures2)
                    utils.plot_figures(the_figures3)

            #self.output = [1,2,3]
        self.output = i.ImageList()

    def energyEqualization(self, fft, AB, direction, factor, symetry, kernel_size = 1):
        """
        cdef:
            int a
            int s
            int x
            float dot
            float pond
            float distance
            float filter_value
            float inverse
            float inverse2
            int p1
            int p2
        """
        assert fft.shape == AB.shape
        center = (AB.shape[0]/2.0, AB.shape[1]/2.0)
        distance = np.linalg.norm(center) * factor
        AB = AB.reshape(-1)
        fft_loop = fft.reshape(-1)
        for x in range(len(fft_loop)):
            a = x // fft.shape[0]
            s = x % fft.shape[1]
    #for a in range(len(A)):
    #    for s in range(len(A[a])):
            #origin_distance = np.linalg.norm([a-center[0], s-center[1]])
            #dot = (a-center[0])*direction[0] + (s-center[1]) * direction[1]
            #if  symetry:
            #    dot=abs(dot)

            #else:
            #    if dot < 0:
            #        dot=0

            #pond = dot/distance
            #ramp = origin_distance/max_distance   
            #if pond > 1:
            #    pond = 1     

            #filter[a][s] = 2 * high_amp * A[a][s]*(1-pond**2) + B[a][s] * pond**2 * high_amp #* (2 - ramp) #* dot / max_distance
            #filter_value = 1 *A[a][s] *(1-pond**2) + B[a][s] * pond**2 *1 #* high_amp #* (2 - ramp) #* dot / max_distance
            #filter_value = 1 *A[x] *(1-pond**2) + B[x] * pond**2 *1 #* high_amp #* (2 - ramp) #* dot / max_distance
            filter_value = AB[x]
            if filter_value < 1:
                inverse2 = 0
                inverse = 0
                for a1 in range(kernel_size):
                    p1 = a+a1-kernel_size//2
                    if p1 >=fft.shape[0]:
                        p1 -= fft.shape[0]
                    
                    for s1 in range(kernel_size):
                        p2 = s+s1-kernel_size//2
                        if p2 >=fft.shape[1]:
                            p2 -= fft.shape[1]

                        energy = np.abs(fft[p1][p2])
                        inverse +=  1
                        inverse2 += 1 / energy

                inverse = inverse / inverse2
                #unit_complex = fft[a][s] / np.abs(fft[a][s])
                #fft[a][s] = unit_complex * ( (1-filter_value) * inverse + np.abs(fft[a][s]) * (filter_value) )#* high_amp #* (2 - ramp) #* dot / max_distance
                unit_complex = fft_loop[x] / np.abs(fft_loop[x])
                fft[a,s] = unit_complex * ( (1-filter_value) * inverse + np.abs(fft_loop[x]) * (filter_value) )#* high_amp #* (2 - ramp) #* dot / max_distance
        
        return fft

    def filterPonderation(self, A, B, direction, factor, symetry):
        assert A.shape == B.shape
        filter = np.zeros_like(A, dtype=float)
        center = (A.shape[0]/2.0, A.shape[1]/2.0)
        distance = np.linalg.norm(center) * factor
        for a in range(len(A)):
            for s in range(len(A[a])):
                #origin_distance = np.linalg.norm([a-center[0], s-center[1]])
                dot = (a-center[0])*direction[0] + (s-center[1]) * direction[1]

                if  symetry:
                    dot=abs(dot)

                else:
                    if dot < 0:
                        dot=0

                pond = dot/distance
                #ramp = origin_distance/max_distance   
                if pond > 1:
                    pond = 1     

                #filter[a][s] = 2 * high_amp * A[a][s]*(1-pond**2) + B[a][s] * pond**2 * high_amp #* (2 - ramp) #* dot / max_distance
                #filter[a][s] =  A[a][s]*pond *(1-pond)**2 + B[a][s] * pond**2 *(1-pond)**2  #* high_amp #* (2 - ramp) #* dot / max_distance
                filter[a][s] = 1 *A[a][s] *(1-pond) + B[a][s] * pond *1 #* high_amp #* (2 - ramp) #* dot / max_distance

        return filter

    def filterImage(self, img, filter):
        print("shape ", img.shape)
        fft_img = utils.fourierTransform(img)
        fft_img*=filter
        img_reco = utils.inv_FFT_all_channel(fft_img)
        #return np.abs(img_reco.real)
        return np.abs(img_reco.real).astype(np.uint8), fft_img

    def gaussianResponse(self, x, y, factorx, factory):
        return math.exp(-( x**2/(2*factorx)+y**2/(2*factory) ))#1/( 1 + x**2/(factorx**2)+y**2/(factory**2) )

    def getSquareFilter(self,shape):
        square_filter = utils.draw_square(shape, 1)
        return square_filter

    def getGaussianFilter(self, shape):
        gaussian = np.zeros(shape,dtype=float)
        center = (shape[0]/2.0, shape[1]/2.0)
        gaussian_factor_x = center[1] * self.std_x
        gaussian_factor_y = center[0] * self.std_y

        for a in range(len(gaussian)):
            for s in range(len(gaussian[a])): # 1 is x, 0 is y, a is y, s is x 
                coord_x = (a-center[0])*self.direction_pass[0] + (s-center[1]) * self.direction_pass[1]
                coord_y = (a-center[0])*self.direction_filter[0] + (s-center[1]) * self.direction_filter[1]

                #gaussian[a][s] = self.gaussianResponse(s - center[1], a - center[0], gaussian_factor_x * gaussian.shape[1], gaussian_factor_y * gaussian.shape[0])
                gaussian[a][s] = self.gaussianResponse(coord_x, coord_y, gaussian_factor_x * gaussian.shape[1], gaussian_factor_y * gaussian.shape[0])

        return gaussian

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def equalizationFilter(self, fft, direction_pass, direction_filter,  ponderation_distance_factor, symmetry,  kernel_size):
        #A = np.zeros(shape,dtype=float).reshape(-1)
        #B = self.getSquareFilter(shape).reshape(-1)

        #cdef float coordx, coordy, distance, dot, pond,  gaussian, square, inverse, inverse2
        #cdef Py_ssize_t a, s, minx, maxx, miny, maxy, offset

        shape = fft.shape
        center = (shape[0]/2.0, shape[1]/2.0)
        gaussian_factor_x = center[1] * self.std_x
        gaussian_factor_y = center[0] * self.std_y
        distance = np.linalg.norm(center) * ponderation_distance_factor

        filter = np.zeros_like(fft, dtype=float)
        #filter_reshape = filter.reshape(-1)
        #for x in range(len(fft_loop)):
        #    a = x // shape[0]
        #    s = x % shape[1]
        
        offset = int((shape[0] * (1 - self.square_proportion)) * 0.5)

        #cdef float[:,:] direction_pass = direction_pass_in
        
        
        absolutes = np.abs(fft)
        for a in range(shape[0]):
            for s in range(shape[1]):
                coord_x = (a-center[0])*direction_pass[0] + (s-center[1]) * direction_pass[1]
                coord_y = (a-center[0])*direction_filter[0] + (s-center[1]) * direction_filter[1]
                gaussian = self.gaussianResponse(coord_x, coord_y, gaussian_factor_x * shape[1], gaussian_factor_y * shape[0])
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
                filter[a][s] = filter_value
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
                        
                    sl1 = absolutes[miny:maxy,minx:maxx]
                    sl = sl1.reshape(-1)

                    for i in range(len(sl)):
                        energy = sl[i] #sl[i].real**2 + sl[i].imag**2
                        inverse +=  1
                        inverse2 += 1/ energy
                        
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
                    if inverse2 != 0:
                        inverse = len(sl) / inverse2
                    
                    unit_complex = fft[a][s]  / absolutes[a][s]
                    fft[a][s] = unit_complex * ( (1-filter_value) * inverse + absolutes[a][s] * (filter_value) )#* high_amp #* (2 - ramp) #* dot / max_distance
                    #fft_loop[x] = fft_loop[x] / np.abs(fft_loop[x]) * ( (1-filter_value) * inverse + np.abs(fft_loop[x]) * (filter_value) )#* high_amp #* (2 - ramp) #* dot / max_distance
        #fft /= absolutes
        return fft, filter
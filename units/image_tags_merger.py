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
import os
import ntpath
#import psyco
#psyco.full()

from units.jit_optimizations import image_tags_merger

class Unit(unit.Unit):
    def __init__ (self, context, unit_config= None, input_map_id_list = None):
        unit.Unit.__init__(self, context, unit_config, input_map_id_list)
        print("__init__ Image Tags Merger")
        self.df_list = self.getInputOrDefault(type(pandas.DataFrame()), []) #inputs ar always lists, should be a struct with meta data from the sender...
        self.image = self.getConfigOrDefault('image', 'image') #column name for image path in the dataframe
        self.image_name = self.getConfigOrDefault('image_name', 'image_name') #column name for image path in the dataframe
        self.meta_data = self.getConfigOrDefault('meta_data', 'meta_data') #column name for image path in the dataframe
        self.category = self.getConfigOrDefault('category', 'category') #column for image category
        self.domain = self.getConfigOrDefault('domain', ['fourier']) # fourier, spatial
        self.ponderation = self.getConfigOrDefault('ponderation', [['direct']]) #direct, average, inverse
        self.max = self.getConfigOrDefault('max', sys.maxsize)
        self.one_ponderation_per_iteration = self.getConfigOrDefault('one_ponderation_per_iteration', True)
        for domain in self.domain:
            if domain not in ['fourier', 'spatial']:
                print("Image Tags Merger domain error " + domain)
                pass
        
        for ponderation_arr in self.ponderation:
            for ponderation in ponderation_arr:
                if ponderation not in ['direct', 'inverse', 'sum']:
                    print("Image Tags Merger ponderation error " + ponderation)
                    pass
        
        assert len(self.ponderation) == len(self.domain)
        
    def run(self):
        print("image tags merger")
        new_width = self.affination
        new_height = self.affination
        if self.df_list is not None and len(self.df_list) > 0:            
            for df in self.df_list: #merges against the same dataframe
                samples = len(df.index)
                print("loaded samples ", samples)
                df_category_list = None
                if self.category != "":
                    df_category_list = [d.sample(frac=1) for _, d in df.groupby([self.category])] #the output is the same in any order, but the output order not... not so useful...
                                    
                else:
                    df_category_list = [df.sample(frac=1)]
                
                for i in range(len(df_category_list)):
                    size = len(df_category_list[i].index) 
                    #print("total_samples: ", size * (size -1) * 0.5)
                    current_index = 0
                    count=0
                    for index, row in df_category_list[i].iloc[0:].iterrows():
                        if count >= self.max:
                            break

                        img1 = utils.getImageFromFile(row[self.image], new_width, new_height, True)
                        img1_max = np.max(img1)
                        ponderation_current_index = 0
                        current_index += 1 #indexes are not correlated with the dataframe order,
                        for index2, row2 in df_category_list[i].iloc[current_index:].iterrows():
                            #print("the indexes : ", index, " - ", index2)
                            img2 = utils.getImageFromFile(row2[self.image], new_width, new_height, True)
                            new_width_merge = int(np.round(np.sqrt(img1.shape[1] * img2.shape[1])))
                            new_height_merge = int(np.round(np.sqrt(img1.shape[0] * img2.shape[0])))
                            img3 = utils.resizeImage(img1, new_width_merge, new_height_merge)
                            img2 = utils.resizeImage(img2, new_width_merge, new_height_merge)

                            fft_A = utils.fourierTransform(img3)
                            fft_B = utils.fourierTransform(img2)
                            fft_merge = None
                            img_merge = None
                            for idm in range(ponderation_current_index, len(self.domain)):
                                domain = self.domain[idm]
                                if domain == 'fourier':
                                    for ponderation in self.ponderation[ponderation_current_index]:
                                        img_merge = image_tags_merger.fourierMerge(ponderation, fft_A, fft_B, img1_max)
                                        fft_merge=None
                                        """
                                        if ponderation == 'direct':
                                            fft_merge = (fft_A * np.abs(fft_A)  + fft_B * np.abs(fft_B)) / (np.abs(fft_A) + np.abs(fft_B))

                                        elif ponderation == 'inverse':
                                            fft_merge = (fft_A * 1/np.abs(fft_A)  + fft_B * 1/np.abs(fft_B)) / (1/np.abs(fft_A) + 1/np.abs(fft_B))

                                        elif ponderation == 'sum':
                                            fft_merge = fft_A + fft_B 
                                        
                                        else:
                                            print("ponderation error")
                                            assert 0
                                        
                                        img_merge = np.abs(utils.inv_FFT_all_channel(fft_merge))
                                        img_merge= (255 * img_merge/np.max(img_merge)).astype(np.uint8)
                                        """
                                        self.writeOutput(domain, ponderation, row, row2, img_merge, img2, img3, fft_A, fft_B, fft_merge)
                                        count +=1

                                elif domain == 'spatial':
                                    for ponderation in self.ponderation[ponderation_current_index]:
                                        img_merge = image_tags_merger.spatialMerge(ponderation, img3, img2)
                                        """
                                        if ponderation == 'direct':
                                            img_merge = (img3 * img3  + img2 * img2) / (img3 + img2)

                                        elif ponderation == 'inverse':
                                            img_merge = 2 / (1/img3 + 1/img2)

                                        elif ponderation == 'sum':
                                            img_merge = (img3 + img2) / 2 
                                        
                                        else:
                                            print("ponderation error")
                                            assert 0
                                        """
                                        self.writeOutput(domain, ponderation, row, row2, img_merge.astype(np.uint8), img2, img3, fft_A, fft_B, None)
                                        count +=1

                                if self.one_ponderation_per_iteration:
                                    ponderation_current_index +=1
                                    ponderation_current_index %= len(self.domain)
                                    break

                                else:
                                    pass
                                
                            if count >= self.max:
                                break

            return True
        
        return False

    def writeOutput(self, domain, ponderation, row, row2, img_merge, img2, img3, fft_A, fft_B, fft_merge):
        if self.write_output_to_disk:
            meta_data_list1 = row[self.meta_data]
            meta_data_list2 = row2[self.meta_data]
            row2_name, ext_img = os.path.splitext(row2[self.image_name])
            path = self.getPathOutputForFile(row[self.image], domain+"_"+ponderation+"-"+row2_name+"-")
            #imageio.imwrite(uri=path, im=equalized[0:500, 500:1024])  
            imageio.imwrite(uri=path, im=img_merge) 
            for entry in meta_data_list1:
                output_path = self.getPathOutputForFile(entry, "")
                dirname = os.path.dirname(output_path)
                filename = ntpath.basename(output_path)
                name_meta1, ext = os.path.splitext(filename)                               
                with open(entry) as first:
                    for entry2 in meta_data_list2:
                        dirname2 = os.path.dirname(entry2)
                        filename2 = ntpath.basename(entry2)
                        name_meta2, ext2 = os.path.splitext(filename2)                               
                        with open(entry2) as second:
                            self.createPath(dirname)
                            with open(dirname+"/"+ domain+"_"+ponderation+"-"+name_meta2+"-"+name_meta1 +ext, 'w') as out_file:
                                out_file.write(first.read()) 
                                out_file.write("\n")
                                out_file.write(second.read()) 
            #imageio.imwrite(uri=path+".png", im=equalized)  

        else:
            the_figures2 = [utils.InputPFF(img2, row2[self.image_name]), utils.InputPFF(img3, row[self.image_name], 0, 0),  utils.InputPFF(img_merge, "merge", 0, 0)]
            utils.plot_figures(the_figures2)    
            if fft_merge is not None:
                the_figures3 = [utils.InputPFF(utils.fft_img_values(np.abs(fft_B)), "fft_B"), utils.InputPFF(utils.fft_img_values(np.abs(fft_A)), "fft_A"),  utils.InputPFF(utils.fft_img_values(np.abs(fft_merge)), " merge fft")]
                utils.plot_figures(the_figures3)
                                    
    #receive AB set and A, returns B set... 
    def merge_difference_cosine(AB, A, mode = 0, absolute = True, truncate_in = True, cosine_90_vs_0= 2):
        # 1/ (j * wo * n) -- for 2d might be 1 /j(wn * n + wm * m)
        out = np.zeros_like(AB,dtype=complex)
        for n in range(AB.shape[0]):
            for m in range(AB.shape[1]):        
                absA =np.abs(A[n][m])
                absAB = np.abs(AB[n][m])
                cosine = 1
                if absA * absAB > 0.1:    
                    if cosine_90_vs_0 == 1:
                        cosine = 1-np.abs(AB[n][m].real * A[n][m].real + AB[n][m].imag * A[n][m].imag) / (absA *absAB)
                        #if cosine > 0:
                        #    cosine = 1 - cosine
                        #else:
                        #    cosine = -1 -cosine
                                          
                    elif cosine_90_vs_0 == 0:
                        cosine = (AB[n][m].real * A[n][m].real + AB[n][m].imag * A[n][m].imag) / (absA *absAB)
                    
                if absolute:
                    cosine = np.abs(cosine)
            
                truncate_loop = False
                if truncate_in:
                    if absAB - absA <0:
                        out[n][m] = 0
                        continue
                    
            
                if mode == 0:
                    out[n][m] = (AB[n][m] - A[n][m])*cosine# * (np.abs(cosine)) #np.dot(np.angle())# / np.abs(AB[n][m] + A[n][m])
            
                elif mode == 1:
                    out[n][m] = (absAB - absA) * cosine 
            
                elif mode == 2:
                    out[n][m] = (absAB - absA) * cosine * AB[n][m]
            
                elif mode == 3:
                    out[n][m] = np.abs(AB[n][m] - A[n][m]) * cosine
                
                elif mode == 4:
                    out[n][m] = np.abs(AB[n][m] - A[n][m]) * cosine * AB[n][m]
            
                elif mode == 5:
                    out[n][m] = (AB[n][m] - cosine * AB[n][m]) #difference without change the phase...
            
                elif mode == 5:
                    out[n][m] = (AB[n][m] - absAB * cosine * A[n][m] / absA) #perpendicular difference...
                
                elif mode == 6:
                    out[n][m] =  cosine 
                    
                
        return out

    def merge_division(AB, B, scalar = 1, A = None, sqrt = True):
        if A == None:
            return scalar * AB / B
    
        else:
            if sqrt:
                return scalar * np.sqrt(AB * A/B)

            else:
                return scalar * AB * A / B 

            

    #receive A and B sets and return the intersection between them...
    def merge_intersection(A, B, scalar = 1, absolute = False, mode = 0):
        out = np.zeros_like(A,dtype=complex)
        for n in range(A.shape[0]):
            for m in range(A.shape[1]):       
                absA =np.abs(A[n][m])
                absB = np.abs(B[n][m])
                cosine = 1
                if absA * absB > 0.001:
                    cosine = (B[n][m].real * A[n][m].real + B[n][m].imag * A[n][m].imag) / (absA *absB)
            
                if absolute:
                    cosine = np.abs(cosine)        
                #sqrt = np.sqrt(A[n][m]*B[n][m]) * cosine  
                if mode == 1:
                    out[n][m] = cosine * ((A[n][m])) * scalar             

                elif mode == 2:
                    out[n][m] = cosine * ((B[n][m])) * scalar             

                elif mode == 3:         
                    out[n][m] = cosine * ((A[n][m] + B[n][m])) * scalar       
            
                elif mode == 4:
                    out[n][m] = cosine * np.sqrt(A[n][m]*B[n][m]) * scalar        
            
                elif mode == 5:
                    out[n][m] = np.sqrt(A[n][m]*B[n][m]) * scalar        
                
                else:
                    out[n][m] = cosine * scalar   
            
        return out;

    def merge_product_intersection(A, B):
        out = np.zeros_like(A,dtype=complex)
        for n in range(A.shape[0]):
            for m in range(A.shape[1]):       
                absA =np.abs(A[n][m])
                absB = np.abs(B[n][m])
                cosine = 0
                if absA * absB > 0.1:
                    cosine = (B[n][m].real * A[n][m].real + B[n][m].imag * A[n][m].imag) / (absA *absB)

                #sqrt = np.sqrt(A[n][m]*B[n][m]) * cosine      
                out[n][m] = cosine * ((B[n][m] * A[n][m]))              
     
        return out;
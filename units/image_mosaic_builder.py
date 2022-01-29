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
import random
import utils
#import psyco
#psyco.full()

class Unit(unit.Unit): #looks into folders and creates the same structure but splitting the contents in the folders. if the file has liked files by name and path a parallel folder is defined
    def __init__ (self, context, unit_config= None, input_map_id_list = None):
        unit.Unit.__init__(self, context, unit_config, input_map_id_list)
        self.df_list = self.getInputOrDefault(type(pandas.DataFrame()), []) #inputs ar always lists, should be a struct with meta data from the sender...
        self.foreground_categories = self.getConfigOrDefault('foreground_categories', []) #feature category and tag for tag file [['fb', 0]]...
        self.background_categories = self.getConfigOrDefault('background_categories', []) #background category
        self.total_compositions = self.getConfigOrDefault("total_compositions", "total_compositions")
        self.image = self.getConfigOrDefault("image", "image")
        self.category = self.getConfigOrDefault("category", "category")
        self.feature_category_name = [self.foreground_categories[i] for i in range(len(self.foreground_categories)) ] # the tag will be the list index..
        
        self.foreground_proportions_per_composition = self.getConfigOrDefault("foreground_proportions_per_composition",[]) #float percetage between 0 and 1 

    def run(self):
        """
        new output path is the unit output path plus de model group number and inside the splitted model
        #take every image and put inside a group, split the number of samples in groups of desired sizes
        """
        if self.df_list is not None and len(self.df_list) > 0:
            for df in self.df_list: #merges against the same dataframe
                features_df =[]
                background_df = []
                for i in range(len(self.foreground_categories)): 
                    features_df.append(df[df[self.category]==self.foreground_categories[i]])
                
                for i in range(len(self.background_categories)): 
                    background_df.append(df[df[self.category]==self.background_categories[i]])
                
                background_df_index = 0
                background_index = 0
                
                features_df_indexes = [0 for x in range(len(features_df))]
                for i in range(self.total_compositions):
                    bg_df = background_df[background_df_index % len(background_df)]
                    row_bg = bg_df.iloc[background_index % len(bg_df.index)] 
                    background_index+=1
                    #get background
                    #get background size
                    #get area proportion(number of pixels that can be filled with features)
                    #fill until you pass the proportions, doesnt matter how much
                    #write the tags with the list index as yolo tag
                    bg_img = utils.getImageFromFile(row_bg[self.image], 0, 0, True)
                    composition = bg_img.copy() #np.ones_like(bg_img)*128
                    total_area = bg_img.shape[0] * bg_img.shape[1]
                    foreground_proportions = []
                    for i in range(len(self.foreground_proportions_per_composition)):
                        foreground_proportions.append(self.foreground_proportions_per_composition[i] * total_area)

                    for i in range(len(features_df)):
                        df = features_df[i]
                        start = features_df_indexes[i]
                        random.seed(round(time.time() * 1000))
                        acc = 0
                        for index, row_fg in df.iloc[start%len(df.index):].iterrows():
                            features_df_indexes[i]+=1
                            fg_img = utils.getImageFromFile(row_fg[self.image], 0, 0, True)
                            y = random.randrange(bg_img.shape[0])
                            x = random.randrange(bg_img.shape[1])
                            from_y = checkInterval(y - fg_img.shape[0] // 2, bg_img.shape[0])
                            from_x = checkInterval(x - fg_img.shape[1] // 2, bg_img.shape[1])
                            to_y = checkInterval(y + fg_img.shape[0] // 2, bg_img.shape[0])
                            to_x = checkInterval(x + fg_img.shape[1] // 2, bg_img.shape[1])
                            
                            coords= []
                            interval_y = to_y-from_y
                            interval_x = to_x-from_x
                            coords.append([from_y, to_y, from_x, to_x])
                            orig_y = fg_img.shape[0] // 2 - (y - from_y)
                            orig_x = fg_img.shape[1] // 2 - (x - from_x)
                            coords.append([ orig_y, orig_y+interval_y, orig_x, orig_x+interval_x])
                            #print(coords)
                            """
                            if from_y > to_y:
                                if from_x > to_x:
                                    coords = []
                                    coords.append([from_y, composition.shape[0], from_x, composition.shape[1]])
                                    coords.append([0,composition.shape[0] - from_y, 0, composition.shape[1] - from_x]) 
                                    coords.append([0, to_y, 0, to_x])
                                    coords.append([0,composition.shape[0] - from_y, fg_img.shape[1] - to_x, fg_img.shape[1]])

                                    #composition[from_y:, from_x:] = 1/( 1/fg_img[:composition.shape[0] - from_y, :composition.shape[1] - from_x] + 1/composition[from_y:,from_x:])               
                                    #composition[:to_y, :to_x] = 1/( 1/fg_img[:composition.shape[0] - from_y:, fg_img.shape[1] - to_x:] + 1/composition[:to_y, :to_x]) 
                                
                                else:
                                    coords = []
                                    coords.append([from_y, composition.shape[0], from_x,to_x])
                                    coords.append([0, composition.shape[0] - from_y, 0, to_x-from_x]) 
                                    coords.append([0, to_y, from_x, to_x])
                                    coords.append([fg_img.shape[0] - to_y, fg_img.shape[0], 0, to_x-from_x])

                                    #composition[from_y:, from_x:to_x] = 1/( 1/fg_img[:composition.shape[0] - from_y, :to_x-from_x] + 1/composition[from_y:,from_x:to_x])               
                                    #composition[:to_y, from_x:to_x] = 1/( 1/fg_img[fg_img.shape[0] - to_y:, :to_x-from_x] + 1/composition[:to_y, from_x:to_x]) 
                                
                            else:
                                if from_x > to_x:
                                    coords = []
                                    coords.append([from_y,to_y, from_x,composition.shape[1]])
                                    coords.append([0,to_y-from_y, 0, composition.shape[1] - from_x]) 
                                    coords.append([from_y,to_y, 0,to_x])
                                    coords.append([0, to_y-from_y, fg_img.shape[1] - to_x, fg_img.shape[1]])

                                    #composition[from_y:to_y:, from_x:] = 1/( 1/fg_img[:to_y-from_y, :composition.shape[1] - from_x] + 1/composition[from_y:to_y,from_x:])               
                                    #composition[from_y:to_y, :to_x] = 1/( 1/fg_img[:to_y-from_y, fg_img.shape[1] - to_x:] + 1/composition[from_y:to_y, :to_x]) 
                                
                                else:
                                    coords = []
                                    coords.append([from_y, from_y + fg_img.shape[0], from_x, from_x+fg_img.shape[1]])
                                    coords.append([0,fg_img.shape[0], 0,fg_img.shape[1]]) 
                                    #composition[from_y:from_y +fg_img.shape[0], from_x:from_x+fg_img.shape[1]] = 1/( 1/fg_img[:,:] + 1/composition[from_y:from_y +fg_img.shape[0],from_x:from_x+fg_img.shape[1]] )  
                            """
                            composition = self.sliceMerge(coords, composition, fg_img)
                            acc += fg_img.shape[0] * fg_img.shape[1]
                            if acc > self.foreground_proportions_per_composition[i] * total_area:
                                break
                    
                    #composition = self.fourierMerge(composition, bg_img)
                    path = self.getPathOutputForFile(row_bg[self.image], "", "")
                    imageio.imwrite(uri=path, im=composition)  
                    
    def inverseMerge(self, composition, fg_img):
        return 2/(1/composition + 1/fg_img)

    def directMerge(self, composition, fg_img):
        return (fg_img.astype(np.float)**2 + composition.astype(np.float)**2)/(fg_img + composition)               
        
    def sliceMerge(self, coords, composition, fg_img):
        for i in range(0,len(coords), 2):
            fore = fg_img[coords[i+1][0]:coords[i+1][1], coords[i+1][2]:coords[i+1][3]] 
            comp = composition[coords[i][0]:coords[i][1], coords[i][2]:coords[i][3]]
            #mix = mix[:, :, 0] 
            maxv = np.max(comp)
            #mix = self.inverseMerge(comp, fore)
            mix = self.fourierMerge(comp, fore)

            #composition[coords[i][0]:coords[i][1], coords[i][2]:coords[i][3]] = (mix[:, :]).astype(np.uint8)
            composition[coords[i][0]:coords[i][1], coords[i][2]:coords[i][3]] = (mix[:, :, 0] * maxv).astype(np.uint8)

            #composition[:to_y, :to_x] = 1/( 1/fg_img[:composition.shape[0] - from_y:, fg_img.shape[1] - to_x:] + 1/composition[:to_y, :to_x]) 
        return composition

    def fourierMerge(self, composition, bg_img):
        fft_A = utils.fourierTransform(composition)
        fft_B = utils.fourierTransform(bg_img)
        fft_merge = (fft_A * np.abs(fft_A)  + fft_B * np.abs(fft_B)) / (np.abs(fft_A) + np.abs(fft_B))
        img_merge = np.abs(utils.inv_FFT_all_channel(fft_merge))
        img_merge /= np.max(img_merge)
        return img_merge

def checkInterval(x, extend):
    if x<0:
        return 0

    if x >= extend:
        return extend-1

    return x

def checkIntervals(intervals, extend):
    for i in range(0,len(intervals), 2):
        if intervals[i+1] - intervals[i] < 0:
            intervals[i] += (intervals[i+1] - intervals[i]) 

        if intervals[i+1] - intervals[i] >= extend:
            intervals[i+1] -= extend - intervals[i+1] - intervals[i]

    return intervals
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
from units.jit_optimizations import model_splitter
#import psyco
#psyco.full()

class Unit(unit.Unit): #looks into folders and creates the same structure but splitting the contents in the folders. if the file has liked files by name and path a parallel folder is defined
    def __init__ (self, context, unit_config= None, input_map_id_list = None):
        unit.Unit.__init__(self, context, unit_config, input_map_id_list)
        print("__init__ Image Tags Merger")
        self.df_list = self.getInputOrDefault(type(pandas.DataFrame()), []) #inputs ar always lists, should be a struct with meta data from the sender...
        self.group_size_from = self.getConfigOrDefault("group_size_from", 100)
        self.group_size_to = self.getConfigOrDefault("group_size_to", 1000)
        self.group_num = self.getConfigOrDefault("group_num", 10)
        self.file_path = self.getConfigOrDefault("file_path", "image")
        self.meta_data = self.getConfigOrDefault("meta_data", "meta_data")
        self.category = self.getConfigOrDefault("category", "category")

    def run(self):
        """
        new output path is the unit output path plus de model group number and inside the splitted model
        #take every image and put inside a group, split the number of samples in groups of desired sizes
        """
        if self.df_list is not None and len(self.df_list) > 0:
            set_offset = 0
            for df in self.df_list: #merges against the same dataframe
                df_category_list = [d for _, d in df.groupby([self.category])]
                #print("total length", len(df.index))
                total_samples = len(df.index)
                proportions = []
                for i in range(len(df_category_list)):
                    #print("category total length", len(df_category_list[i].index))
                    proportions.append(len(df_category_list[i].index)/total_samples)

                iteration = 1
                set_sizes = []
                set_increments = []
                """
                while total_samples > 0:
                    total_samples -= int(self.group_size_to // iteration)
                    if total_samples<self.group_size_from or total_samples < len(df_category_list):
                        total_samples += int(self.group_size_to // iteration)
                        set_sizes.append(int(total_samples))
                        break

                    else:
                        set_sizes.append(self.group_size_to // iteration)
                    
                    if self.group_size_to // (iteration + 1) <= self.group_size_from:
                        iteration = 1

                    else:
                        iteration += 1
                """        
                set_sizes = model_splitter.computeModelSplits(total_samples, self.group_size_to, self.group_size_from, len(df_category_list))
                for i in range(len(df_category_list)):
                    set_index = 0
                    set_increments = [0 for i in range(len(set_sizes))]
                    for index, row in df_category_list[i].iterrows():
                        file_path = row[self.file_path]  
                        #print(file_path)
                        set_index = model_splitter.nextIndex(i, int(set_index), tuple(set_sizes), tuple(proportions), tuple(set_increments))   
                        """
                        while set_index < len(set_sizes) and np.floor(set_sizes[set_index]*proportions[i]) <= set_increments[set_index]:
                            set_index += 1
                        
                        if set_index >= len(set_sizes):
                            set_index = 0
                            while set_index < len(set_sizes) and np.floor(set_sizes[set_index]*proportions[i]) <= set_increments[set_index]:
                                set_index += 1

                            if set_index >= len(set_sizes):
                                set_index = 0
                        """
                        #save file into folder
                        self.writeOutput(set_index+set_offset, row[self.file_path], row[self.meta_data])
                        #set_sizes[set_index] -=1  
                        set_increments[set_index] +=1                           
                        set_index +=1
                
                for set_index in range(len(set_sizes)):
                    self.copyFiles(str(set_index+set_offset))
                    self.copyFolders(str(set_index+set_offset))

                set_offset += len(set_sizes)
        
        return True 
        
    def writeOutput(self, set_index, file_name, metas):
        with open(file_name, 'rb') as in_file:
            out_path = self.getPathOutputForFile(file_name, "", str(set_index)) # mal.....
            #print(out_path)
            with open(out_path, 'wb') as out_file:
                out_file.write(in_file.read()) 

        for entry in metas:
            with open(entry, 'rb') as in_file:
                out_path = self.getPathOutputForFile(entry, "", str(set_index))
                #print(out_path)
                with open(out_path, 'wb') as out_file:
                    out_file.write(in_file.read()) 
   
        """
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
        """
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
from multiprocessing import Process, Pipe
#from img2vec_pytorch import Img2Vec
#from .img2vec.img2vec_pytorch.img_to_vec import Img2Vec
import utils
import os, sys
import pandas as pd
import pylab as pl
import numpy as np
import scipy.stats as stats
from units.jit_optimizations import image_embedder
from numba import njit, jit


class Unit(unit.Unit):
    def __init__ (self, context, unit_config= None, input_map_id_list = None):
        unit.Unit.__init__(self, context, unit_config, input_map_id_list)
        print("__init__ Image Embedder")
        self.df_list = self.getInputOrDefault(type(pandas.DataFrame()), [])
        self.plot_output = self.getConfigOrDefault('plot_output', False)
        self.image = self.getConfigOrDefault('image', 'image')
        self.what_embedding = self.getConfigOrDefault('embedding', 'power_harmonics')
        self.cuda = self.getConfigOrDefault('cuda', False)
        self.load_embedding = self.getConfigOrDefault('load_embedding', False)

    def run(self):
        self.output = []
        count = 0
        file_present = False
        if self.load_embedding:
            if self.what_embedding == 'power_harmonics':
                out_path = self.getPathOutputForFile("embedding_power_spectrum.csv", str(count)+"_") # mal.....
                file_present = os.path.isfile(out_path)
                if file_present:
                    self.output.append(pandas.read_csv(out_path))

        if file_present is False:
            if self.df_list is not None and len(self.df_list) > 0:
                for df in self.df_list: #merges against the same dataframe
                    if self.what_embedding == 'power_harmonics':
                        out_df = self.startPowerHarmonicsEmbedding(df)
                        if out_df is not None: 
                            if self.write_output_to_disk:
                                #if more than one dataframe the name will be ovewritten..
                                out_path = self.getPathOutputForFile("embedding_power_spectrum.csv", str(count)+"_") # mal.....
                                file_present = os.path.isfile(out_path)
                                if not file_present: 
                                    out_df.to_csv(out_path, index=False)
                                
                                elif self.overwrite_output:
                                    os.remove(out_path)
                                    out_df.to_csv(out_path, index=False)
                            
                            self.output.append(out_df)
    
                        count += 1
                            
                    #elif what_embedding == embedder.resnet_50:
                    #    self.resnet50_df = self.startNNEmbedding()
                    #    if self.resnet50_df is not None:
                    #        self.resnet50_df.to_csv(self.save_path+"/embedding_resnet_50.csv")
                    #        self.current_df = self.resnet50_df

                    #else:
                    #    self.resnet50_df = self.startNNEmbedding()
                    #    self.power_harmonics_df = self.startPowerHarmonicsEmbedding()
                    #    if self.power_harmonics_df is not None:
                    #        self.power_harmonics_df.to_csv(self.save_path+"/embedding_power_spectrum.csv")
                    #    
                    #    if self.resnet50_df is not None:
                    #        self.resnet50_df.to_csv(self.save_path+"/embedding_resnet_50.csv")
                        
                    #    self.current_df = self.power_harmonics_df
        
        return len(self.output) > 0

    def singleImageEmbedding(self, img, df, convert= True):
        if self.what_embedding == 'power_harmonics':
            power_harmonics_df = self.startPowerHarmonicsEmbedding(df, img)
            return power_harmonics_df

        #elif what_embedding == embedder.resnet_50:
        #    if convert:
        #        img = Image.fromarray(img).convert('RGB')
            
        #    resnet50_df = self.startNNEmbedding(df, img)
        #    return resnet50_df

    def loadEmbedding(self, what_embedding):
        try:
            # Path to be created
            #utils.createPath(self.save_path)
            if what_embedding == 'power_harmonics':
                out_path = self.getPathOutputForFile("embedding_power_spectrum.csv") 
                df= pd.read_csv(out_path,  index_col=0)
                return df
            #elif what_embedding == embedder.resnet_50:
            #    self.resnet50_df = pd.read_csv(self.save_path+"/embedding_resnet_50.csv",  index_col=0)
            #    self.resnet50_df = self.resnet50_df.dropna()
            #    self.current_df = self.resnet50_df

            #else:
            #    self.power_harmonics_df = pd.read_csv(self.save_path+"/embedding_power_spectrum.csv",  index_col=0)
            #    self.power_harmonics_df = self.power_harmonics_df.dropna()
    
            #    self.resnet50_df = pd.read_csv(self.save_path+"/embedding_resnet_50.csv",  index_col=0)
            #    self.resnet50_df = self.resnet50_df.dropna()

            #    self.current_df = self.power_harmonics_df
    
            return self.current_df
    
        except:
            self.buildEmbbeding(what_embedding)
            return self.current_df

        return None
    

    def startPowerHarmonicsEmbedding(self, df, img=None):
        #read a csv dataframe...
        #in_object is a dataframe with images data
        #loopea la imagen entera, dividela en una cuadricula de n slices...
        #el promedio general tiene que ser parecido al promedio de slices...??
        #lo que se aleja del promedio es foreign body?
        #scatter =[] 
        #imap = {'fb':0, 'clean':1}
        #self.affination = 1024
        #arr = np.array([1, 2, 3, 4,5])
        #Unit.test_stats(arr)
        
        #arr = np.array([1, 2, 3, 4])
        #Unit.test_stats(arr)
        
        #arr = np.array([1, 2, 3])
        #Unit.test_stats(arr)
             
        #arr = np.array([1, 2])
        #Unit.test_stats(arr)
        
        columns =  df.columns.tolist() 
        #columns += [ 'height', 'width', 'mode', 'median', 'std', 'direct', 'inverse', 'min', 'max', 'pmode', 'pmedian', 'pstd', 'pdirect', 'pinverse', 'pmin', 'pmax']
        columns += ['aspect', 'direct', 'inverse']

        columns += ["n"+str(x) for x in range(self.affination)]
        data = []
        out_df = pd.DataFrame(columns = columns)
        #print(df.columns)
        num=0
        out_list = []
        count = 0
        now = time.time()
        for index, row in df.iterrows():
            #if Cluster in activation_filters:
            """
            img = getImageFromFile("./cloud.png", 0,0)
            #img =  mpimg.imread(basepath+"/"+row['image'])
            print(img.shape)
            parameters = row['image'].split(".")[-2].split("_")[1:-1]
            print(parameters)
            exposure = float(parameters[0][3::])
            gain = float(parameters[1][1::])
            print(exposure)
            print(gain)
            """
            if count % 1000 == 0:
                print("Embedded: " +str( count))
                print("embedding Ellapsed ", time.time() - now)                                    

            count +=1    
            name = row['image']

            neww = list(row.to_numpy()) + image_embedder.powerHarmonicsEmbeddingForSingleImage(name, self.affination, self.plot_output)
            
            #arr = np.asarray(neww)
            #pl.plot(np.concatenate((kvals, kvals + np.max(kvals))), arr)
            #neww = [height/width, direct, inverse]

            #pl.show()
            #row_df = pd.Series(neww)
            #print(neww)
            out_list.append(neww)
            #out_df.loc[len(out_df.index)]= neww

            #print(df.head())
            #print(df['511'])
            #print(Abins[511])

            #df = df.append(neww,ignore_index=True)
            #df = pd.concat([row_df, df], ignore_index=True)
            #df = df.drop(['category', 'Cluster', 'image'], axis=1)
            #neww +=Abins
            #data.append(neww)
            #print(df.iloc[0].values[3])
            #print(Abins == df.iloc[0].values)

            
            #pl.plot(kvals, df.iloc[0].values[3::])
            #pl.figure(imap[row['category']])
            #pl.loglog(kvals, df.iloc[0].values[3::])
            #pl.xlabel("$k$")
            #pl.ylabel("$P(k)$")
            #pl.tight_layout()
            #pl.savefig(basepath+"/"+row['category']+"_spectrum"+str(num)+".png", dpi = 300, bbox_inches = "tight")
            
            #pl.show()
        out_df = pd.DataFrame(out_list, columns = columns)
        print(out_df.size)
        out_df = out_df.dropna()
        print(out_df.size)
        #self.power_harmonics_df = self.power_harmonics_df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
        return  out_df

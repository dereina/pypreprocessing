
import imageio
import utils
import os
import pandas as pd
import units.unit as unit
import time
from units.jit_optimizations import image_loader_to_dataframe

class Unit(unit.Unit):
    def __init__ (self, context, unit_config= None, input_map_id_list = None): #the data schema of the input map is checked by this Unit
        unit.Unit.__init__(self, context, unit_config, input_map_id_list)
        print("__init__ ImageLoaderToDataframe")
        self.append_by_name_meta_data = self.getConfigOrDefault("append_by_name_meta_data", [])
        self.meta_data_extensions = self.getConfigOrDefault("meta_data_extensions", ['txt'])
        self.image_extensions = self.getConfigOrDefault("image_extensions", ['.bmp', '.png'])
        self.prevent_load_errors = self.getConfigOrDefault("prevent_load_errors", True)
        self.meta_data = self.getConfigOrDefault("meta_data", "meta_data")
        self.image = self.getConfigOrDefault("image", "image")
        self.image_name = self.getConfigOrDefault("image_name", "image_name")
        self.category = self.getConfigOrDefault("category", "category")
        self.size = self.getConfigOrDefault("size", "size")
        self.width = self.getConfigOrDefault("width", "width")
        self.height = self.getConfigOrDefault("height", "height")
        self.folders = self.getConfigOrDefault("folders", [])
        self.columns = [self.category, self.image, self.image_name, self.size, self.width, self.height, self.meta_data]
        self.check_type_error = self.getConfigOrDefault("check_type_error", False)

    def run(self):
        
        self.load_errors = []
        print(os.getcwd())
        #i = 0
        print("going inside man")
        row_list = image_loader_to_dataframe.loadDataframe(tuple(self.image_extensions), self.prevent_load_errors, tuple(self.folders), tuple(self.append_by_name_meta_data), tuple(self.meta_data_extensions), [])
        df = pd.DataFrame(row_list, columns = self.columns)
        print("out ", len(df.index))
        """
        for path in self.folders:
            try:
                for name in utils.getFileNamesFrom(path):
                    #i+=1
                    #if i > 5:
                    #    break

                    #print(context.origin+"/"+path+"/"+name)
                    img = utils.getImageFromFile(path+"/"+name, 0)
                    if img.shape == (1,1,3):
                        self.load_errors.append(path+"/"+name)
                        continue
                    #print(path+"/"+name)
                    #print("path: "+path)
                    #savepath = context.origin+"/"+path+"/"+name
                    #if prepend_category:
                    #    if not name.startswith(path):
                            #print("preprend")
                    #       os.rename(context.origin+"/"+path+"/"+name, context.origin+"/"+path+"/"+path+"-"+name) 
                    #        savepath = context.origin+"/"+path+"/"+path+"-"+name
                    #        name = path+"-"+name

                    #if fix_bmp: #resave images if there is a problem with bmp header that won't load on some libraries but with imageio 
                    #    imageio.imwrite(savepath, img)
                    img_name, imgext = os.path.splitext(name)
                    meta_data_list = []         
                    #for entry in self.append_by_name_meta_data:
                    #    for meta_data in utils.getFileNamesFrom(entry):
                    #        name_meta, ext = os.path.splitext(meta_data)
                    #        if img_name == name_meta:
                                #print(meta_data)
                    #            meta_data_list.append(entry+"/"+meta_data)
                    for entry in self.append_by_name_meta_data:
                        for extension in self.meta_data_extensions: 
                            for filename in utils.getFileIfExistFrom(entry, img_name  + extension):
                                #print(entry+"/"+filename)
                                meta_data_list.append(entry+"/"+filename)

                    neww = [path, path+"/"+name, name, img.shape[0] * img.shape[1], img.shape[1], img.shape[0], meta_data_list]
                    #row_df = pd.Series(neww)
                    df.loc[len(df.index)]= neww
            
            except Exception as e:
                print(e)
                raise
                #print(context.origin+"/"+path+"/"+name)

        print(self.load_errors)
        """
        self.output = df
        return True

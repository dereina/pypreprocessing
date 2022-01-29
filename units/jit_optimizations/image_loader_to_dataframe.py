import imageio
import utils
import os
import pandas as pd
import units.unit as unit
import time
from numba import njit, jit

@jit(cache=False, forceobj = True)
def getMetaData(append_by_name_meta_data, meta_data_extensions):
    out = []
    for entry in append_by_name_meta_data:
        for extension in meta_data_extensions:
            out.append((entry, extension))
    
    return tuple(out)

def printTime(length, now):
    print(length)
    print("Ellapsed: " +str( time.time()-now))

@jit(cache=True)
def loadDataframe(image_extensions, prevent_load_errors, folders, append_by_name_meta_data, meta_data_extensions, load_errors):
    print(folders)
    now = time.time()
    print("eiiii youuuuu ")
    row_list = []
    for i in range(len(folders)):
        path = folders[i]
        print("new path:")
        print(path)
        count = 0
        for name in os.listdir(path):
            if os.path.isfile(os.path.join(path, name)):
                if count % 100 == 0:
                    printTime(count, now)

                    #i+=1
                    #if i > 5:
                    #    break
                img_name, imgext = os.path.splitext(name)
                shape= (1,1)
                if prevent_load_errors:
                    shape = utils.checkImageFromFileGetShape(path+"/"+name)
                    if load_errors is not None:
                        if shape == (1,1,3):
                            load_errors.append(path+"/"+name)
                            continue
                
                else:
                    if imgext not in image_extensions:
                        print("Not a file image: ", name)
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
                
                meta_data_list = []         
                #for entry in self.append_by_name_meta_data:
                #    for meta_data in utils.getFileNamesFrom(entry):
                #        name_meta, ext = os.path.splitext(meta_data)
                #        if img_name == name_meta:
                            #print(meta_data)
                #            meta_data_list.append(entry+"/"+meta_data)
                #for entry in append_by_name_meta_data:
                #    for extension in meta_data_extensions: 
                #        for filename in utils.getFileIfExistFrom(entry, img_name  + extension):
                            #print(entry+"/"+filename)
                #            meta_data_list.append(entry+"/"+filename)
                for (entry, extension) in getMetaData(append_by_name_meta_data, meta_data_extensions):
                    filename = utils.getFileIfExistFrom(entry, img_name + extension)
                    if filename: 
                        meta_data_list.append(entry+"/"+filename)

                
                shape = (1,1)
                neww = [path, path+"/"+name, name, shape[0] * shape[1], shape[1], shape[0], meta_data_list]
                #row_df = pd.Series(neww)
                #print(neww)
                count += 1
                row_list.append(neww)
                #df.append(row_df, ignore_index=True)
                #print(df)
                #df.reset_index(drop=True)
                #df.loc[len(df.index)]= neww
  
        #except Exception as e:
        #    print(e)
        #    raise
            #print(context.origin+"/"+path+"/"+name)
            
    print("loaded: ", len(row_list))
    print(load_errors)
    return row_list
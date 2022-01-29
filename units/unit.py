import os
from distutils.dir_util import copy_tree
from shutil import copyfile
from contextlib import contextmanager
import time
class Unit(object):
    def __init__(self, context, unit_config= None, input_map_id_list = None):
        print("__init__ Unit")
        self.unit_config = unit_config
        self.input_map_id_list = input_map_id_list # an schema provider map...
        self.context = context #application context, to get global affination, root, and global parameters in general...        self.inputs = None
        self.persist_inputs = self.getConfigOrDefault('persist_inputs', False)
        self.persist_unit = self.getConfigOrDefault('persist_unit', False)
        if self.unit_config['id'] in input_map_id_list:
            self.inputs = input_map_id_list[self.unit_config['id']]
            print("found inputs for " + self.unit_config['id'])

        else:
            print("found NOT inputs for " + self.unit_config['id'])
        
        self.affination = self.getConfigOrDefault('affination', context.affination)
        self.output = None #an schema provider map??
        self.write_output_to_disk = self.getConfigOrDefault('write_output_to_disk', False)
        self.overwrite_output = self.getConfigOrDefault('overwrite_output', False)

    @contextmanager    
    def writeOutputToDisk(self):
        try:
            yield #if an error occurs during outputs the error will stop the program from copying anything...
        
        except Exception:
            print("exception during writeOutputToDisk")
            raise

        finally:
            if self.write_output_to_disk:
                self.copyFolders()
                self.copyFiles()

    def createPath(self, path):
        try:
            os.makedirs(path, 0x755 )
            #Path(path).mkdir(parents=True)
            return True

        except Exception as e:
            #print("Create Path Error!", e.__class__, path)
            #print("Next entry.")
            #print()
            paths = path.split("/")
            curdir = os.getcwd()
            for i in range(len(paths)):
                #print(os.getcwd())
                if paths[i] == '':
                    paths[i] = './'

                try:
                    os.mkdir(paths[i])

                except Exception as o:
                    pass
                    #print(o)
                    #print ("Creation of the directory %s failed" % path)

                else:
                    print ("Successfully created the directory %s " % path)
                
                os.chdir(paths[i])

            os.chdir(curdir)

        return False

    def getPathOutputForFile(self, relative_path_to_file, prepend = "", prepend_folder = ""):
        #prepends foder output for this unit, creates the path for the file if needed
        root = self.getConfigOrDefault('output_path', self.unit_config['id'])
        if root != '':
            root += "/"

        path = relative_path_to_file.split("/")

        #print(path[-1])
        file_name = path[-1]
        path = "/".join(path[:-1])
            
        if prepend_folder != "":
            prepend_folder = prepend_folder + "/"
        
        self.createPath(root + prepend_folder + path)
        if path != '':
            path += "/"

        return root + prepend_folder + path + prepend + file_name

    def copyFolders(self, prepend_folder = ""):        
        folders_list = self.getConfigOrDefault('copy_folders', [])
        root =  self.getConfigOrDefault('output_path', self.unit_config['id'])
        if root != '':
            root += "/"

        if prepend_folder != "":
            prepend_folder = prepend_folder + "/"

        for path in folders_list:
            self.createPath(root + prepend_folder + path)
            copy_tree(path, root + prepend_folder + path ) #need control for error, but without control is easy to debug a mistake...
    
    def copyFiles(self, prepend_folder = ""):
        files_list = self.getConfigOrDefault('copy_files', [])
        root = self.getConfigOrDefault('output_path', self.unit_config['id'])
        if root != '':
            root += "/"

        if prepend_folder != "":
            prepend_folder = prepend_folder + "/"

        for file in files_list:
            path = file.split("/") 
            #print(path[-1])
            file_name = path[-1]
            path = "/".join(path[:-1])
            self.createPath(root + prepend_folder + path)
            print("/".join(path))
            #self.createPath("/".join(path[::-1]))
            copyfile(file, root + prepend_folder + path + "/" + file_name) #need control for error, but without control is easy to debug a mistake...

    def getUnitId(self):
        return self.unit_config['id'] 

    def getUnitName(self):
        return self.unit_config['unit'] #could use __file__ or other things to retrieve the current file, anyway a bad configured unit is not going to be instantiated 

    def getConfigOrDefault(self, id, default_config):
        if id in self.unit_config:
            return self.unit_config[id]

        print("default config for " + self.unit_config['id'] + " for " + id)
        self.unit_config[id] = default_config
        return default_config

    def getInputOrDefault(self, typeid, default_input):
        if self.inputs is not None:
            print(typeid)
            if typeid in self.inputs:
                return self.inputs[typeid]
                
        print("default input for " + self.unit_config['id'] + " for " + str(typeid))
        return default_input

    def run(self):
        return False

    def removeInputs(self):
        if not self.persist_inputs:
            if self.unit_config['id'] in self.input_map_id_list: 
                del self.input_map_id_list[self.unit_config['id']]

    def exe(self):
        print("exe Unit: " + str(self.unit_config['id']))
        start = time.time()
        if self.run():
            if self.write_output_to_disk:
                self.copyFolders()
                self.copyFiles()

            self.removeInputs()
            end = time.time()
            print(str(self.unit_config['id']) + " Ellapsed: ")
            print(end - start)
            return True

        end = time.time()
        print(str(self.unit_config['id']) + " Ellapsed: ")
        print(end - start)
        self.removeInputs()
        return False

    def getOutputSchema(self):
        return self.output
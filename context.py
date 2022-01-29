import pickle
import yaml
import os
import units.energy_equalizer as energy_equalizer
import importlib
from contextlib import contextmanager
import utils

class Context(object):
    def __init__(self, config_path = r'config.yaml'):
        with open(r'config.yaml') as file:        
            config = yaml.load(file, Loader=yaml.FullLoader)
            print(config)
            print(os.getcwd())
            #f = load_class("units.energy_equalizer").energy_equalizer()
            #f.run()
            #print(f)
            
            
            self.config = config
            self.root = self.config['root']
            self.save_runs = self.config['save_runs']
            self.affination = self.config['affination']
            self.units = {}
            self.persisted_units = {}
            for i in range(len(self.config['units'])):
                self.units[self.config['units'][i]['id']] = self.config['units'][i]

            print(self.units)

            self.paths = self.config['paths']
            self.trees = self.config['trees']
            print(self.paths)

    def runPaths(self):
        with utils.rememberCwd(self.root):
            print(os.getcwd())
            input_map = {}
            for p in range(len(self.paths)):
                for i in range(len(self.paths[p])):
                    unit_config = self.units[self.paths[p][i]]
                    print(unit_config['id'])
                    unit_lib = None
                    current = None
                    if not unit_config['id'] in self.persisted_units:
                        unit_lib = importlib.import_module('.'+unit_config['unit'], package='units')
                        current = unit_lib.Unit(self, unit_config, input_map)
                    
                    else:
                        current = self.persisted_units[unit_config['id']]
                    
                    if current.exe(): #run the unit
                        print(unit_config['unit'] + " with id " + unit_config['id'] + " returned True")
                   
                    else:
                        print(unit_config['unit'] + " with id " + unit_config['id'] + " returned False")

                    if unit_config['persist_unit'] is True and unit_lib is None:
                        self.persisted_units[unit_config['persist_unit']] = current #the instance is new, store it for future calls

                    #set schema input
                    #the unit finds in the input map the schema input added, if multiple of the same schema, then the schema is an array of the same schema...
                    #no arrays of different schemas, the schema is identified by the map, so if not a same schema provider should be added to manage the schemas...

                    #set inputs from current unit for the next unit
                    output = current.getOutputSchema() #should return an schema provider map...
                    if i+1<len(self.paths[p]) and output is not None:
                        if self.paths[p][i+1] not in input_map:
                            input_map[self.paths[p][i+1]] = {}
                        if type(output) is not type(list()) and type(output) not in input_map[self.paths[p][i+1]]:
                            input_map[self.paths[p][i+1]][type(output)] = []
                        
                        elif type(output) == type(list()) and len(output) > 0 and type(output[0]) not in input_map[self.paths[p][i+1]]:
                            input_map[self.paths[p][i+1]][type(output[0])] = []

                        if type(output) == type(list()):
                            if len(output) > 0:
                                input_map[self.paths[p][i+1]][type(output[0])] += output #casi bien pero no... añade el output con la id del sender, una struct que albergue los dos cosas, la id o ref del unit sender y el schema output de ese sender...
                        
                        else:
                            input_map[self.paths[p][i+1]][type(output)].append(output) #casi bien pero no... añade el output con la id del sender, una struct que albergue los dos cosas, la id o ref del unit sender y el schema output de ese sender...
                        
                        print(type(output))

                    #if type(i) not in input_map:
                    #    input_map[type(output)] = []

                    #input_map[type(output)].append(output) #casi bien pero no...
                    

                    #lo que quieres es algo que vaya con la id de los edges... el id de los edges los tiene que conocer la unit...
                    #el input map guarda los outputs para el siguiente unit con la id del edge de la unit que los recibe... tengo images list saliendo por un edge...

            print(input_map)

    def runTrees(self):
        #build tree and run... create a grph first with some tool for building graphs in python...
        pass
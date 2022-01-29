import units.unit as unit
#import units.jit_optimizations.energy_equalizer as energy_equalizer
import pandas

#import psyco
#psyco.full()

class Unit(unit.Unit):
    def __init__ (self, context, unit_config= None, input_map_id_list = None): #the data schema of the input map is checked by this Unit
        unit.Unit.__init__(self, context, unit_config, input_map_id_list)
        print("__init__ Classifier")
        self.df_list = self.getInputOrDefault(type(pandas.DataFrame()), []) #inputs are always lists, should be a struct with meta data from the sender...

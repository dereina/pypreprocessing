import sys
#sys.path.insert(1, './')

import pickle
import yaml
import os
import units.energy_equalizer as energy_equalizer
import importlib
import context

#import schemas.image_list as i

#i.ImageList()

ctx = context.Context(r'config.yaml')
ctx.runPaths()
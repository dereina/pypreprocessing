from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
from Cython.Distutils import build_ext
import os
import sys
import shutil 
import numpy

folder = "."
if len(sys.argv) >3:
  folder = sys[argv[3]]

elif len(sys.argv) <3:
  print("ARGUMENTS ERROR, put the script direction parameter")
  exit()

direction = 1
try:
  direction = int(sys.argv[2])
except:
  pass

os.chdir(folder)
try:
    os.makedirs("backfiles", 0x755 );

except:
    pass

print("usage")
print("python.exe .\setup.py build_ext 1 #for unbuilding")
print("python.exe .\setup.py build_ext 0 #for building")

if direction == 1:
  #change pyx to py....
  with os.scandir() as i:
      for entry in i:
          if entry.name == os.path.basename(__file__):
            continue
          print(entry.name)
          if entry.is_file() and (entry.name.endswith(".pyd") or entry.name.endswith(".c") or entry.name.endswith(".pyx") or entry.name.endswith(".html")):
              os.remove(entry.name)

          elif entry.is_dir() and entry.name == "build":
              shutil.rmtree(entry.name)

          elif entry.is_dir() and entry.name == "__pycache__":
              shutil.rmtree(entry.name)


  os.chdir("backfiles")
  with os.scandir() as i:
    for entry in i:
          print("jake "+ entry.name)
          if entry.is_file():
              print("moving")
              shutil.move(entry.name, "../"+entry.name)

          elif entry.is_dir() and entry.name == "build":
              shutil.movetree(entry.name, "../"+entry.name) 

  exit()
else:
  #change py to pyx
  ext_modules=[]
  with os.scandir() as i:
      for entry in i:
          if entry.name == os.path.basename(__file__):
            continue
          if entry.is_file() and entry.name.endswith(".py"):
              name = os.path.splitext(entry.name)[0]
              shutil.copy(entry.name, name+'.pyx')
              if entry.name.startswith("__") is True:
                continue #don't add files like __init__.pyx to the extensions...

              print(name + " ---  "+name+'.pyx')
              shutil.move(entry.name, "backfiles/"+entry.name)
              ext_modules.append(Extension(name, [name+'.pyx'], include_dirs = ['.']))
              print(entry.name)
 
  
  """
  ext_modules=[
      Extension("utils",       ["utils.pyx"], include_dirs = ['.']),
      Extension("embedder",         ["embedder.pyx"], include_dirs = ['.']),
      Extension("classifier",         ["classifier.pyx"], include_dirs = ['.']),
      Extension("image_loader",         ["image_loader.pyx"], include_dirs = ['.']),
      Extension("context",         ["context.pyx"], include_dirs = ['.']),
      Extension("model_predictor",         ["model_predictor.pyx"], include_dirs = ['.']),
  ]
  """

  setup(
    name="embedder_classifier",
    ext_modules=cythonize(
    ext_modules, 
    compiler_directives={'language_level' : "3"}) ,  # or "2" or "3str" ,
    include_dirs=[numpy.get_include()],
    cmdclass = {'build_ext': build_ext},
    script_args = ['build_ext'],
    options = {'build_ext':{'inplace':True, 'force':True}}
  )


  #all files in a folder
  #setup(
  #  name = 'embedder_classifier',
  #  cmdclass = {'build_ext': build_ext},
  #  ext_modules = ext_modules, #cythonize(["*.pyx"]),
  #)

  #setup(
    #name = 'embedder_classifier',
  #  cmdclass = {'build_ext': build_ext},
  #  ext_modules = cythonize(["*.pyx"]),
  #)
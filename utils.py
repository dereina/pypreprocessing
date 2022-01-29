import numpy as np
import imageio
from PIL import Image
from skimage import transform, io
from itertools import product
import os, sys
import matplotlib.pyplot as plt
import math 
from scipy import ndimage, misc
from contextlib import contextmanager
import pickle
import time;
from scipy import stats
from pathlib import Path

from numba import njit, jit
import itertools
import gc
@contextmanager
def rememberCwd(chdir = None):
    curdir = os.getcwd()
    try:
        if chdir is not None:
            os.chdir(chdir)

        yield

    finally:
        os.chdir(curdir)

def getTimestamp():
    ts = time.time()
    return ts

def createPath(path):
    try:
        os.makedirs(path, 0x755 );
        #Path(path).mkdir(parents=True)
        return True

    except Exception as e:
        print("Oops!", e.__class__, "occurred.")
        print("Next entry.")
        print()
        paths = path.split("/");
        acc = ""
        with rememberCwd():
            for i in range(len(paths)):
                print(os.getcwd())
                try:
                    os.mkdir(paths[i])
                except Exception as o:
                    print(o)
                    print ("Creation of the directory %s failed" % path)
                else:
                    print ("Successfully created the directory %s " % path)
                
                os.chdir(paths[i])

                
    return False

def makedirs(dir,end):
    try:
        os.makedirs( dir+"/"+end);
        for root, dirs, files in os.walk(dir):  
          for momo in dirs:  
            os.chown(os.path.join(root, momo), 777, 20)
          for momo in files:
            os.chown(os.path.join(root, momo), 777, 20)
    
    except:
        print("error creating folder ")
        pass


def savePkcls(name, data):
    filename = name
    with open(filename,'wb+') as (outfile):
            pickle.dump(data, outfile)
            return True
    #outfile.close()
    return False

def openPkcls(name):      
    filename = name
    try:
        with open(filename,'rb+') as (infile):
            return pickle.load(infile)
    except:
        print("error loading file" + filename)        
    return None

def resizeImage(im, resize, height = 0):
    if(resize != 0):
        try:
            [h, w] = im.shape
        except:
            [h, w, c] = im.shape
        ratio = h / w    
        
        if height != 0:
            ratio = height / resize  
            w = resize
                                
        wi = resize
        if( w > wi or height != 0):
            #print("resizing man!!!!! ", wi, " - joe - " ,ratio * wi)
            im = transform.resize(im, (int(wi * ratio), wi), preserve_range=True)        #basewidth = 300

    return im


def getPillImageFromFile(namein, resize, height = 0):
    try:
        #print(namein)
        img = Image.open(namein)
        if(resize != 0):
            
            w, h = img.size
  
            ratio = h / w    
            if height != 0:
                ratio = height / resize  
                w = resize
                                
            wi = resize
            if( w > wi or height != 0):
                print("resizing man!!!!! ", wi, " - joe - " ,ratio * wi)
                img = img.resize((int(wi * ratio), wi), Image.ANTIALIAS)        #basewidth = 300
     
        return img#im.astype(np.uint8)
        
    except Exception as e:
        print(" ERROR LOADING IMAGE MAN!!!! ", namein)
        print(" ERROR ", e)
        
    return np.zeros([1, 1, 3], dtype=np.int8)   

def getImageFromFile(namein, resize, height = 0, gray=False):
    try:
        #print(namein)
        im = imageio.imread(namein)
        im = resizeImage(im, resize, height)
        if gray:
            try:
                im = (0.2126*im[:,:,0]+0.7152*im[:,:,1]+0.0722*im[:,:,2]).astype(np.uint8)

            except:
                pass

        return im.astype(np.uint8)
        
    except Exception as e:
        print(" ERROR LOADING IMAGE MAN!!!! ", namein)
        print(" ERROR ", e)
        
    return np.zeros([1, 1, 3], dtype=np.int8)   

@jit(cache=True, nopython=True)
def mosaicImage(im, affination):
    out = np.zeros((affination, affination)).astype(np.uint8)
    #im = (0.2126*im[:,:,0]+0.7152*im[:,:,1]+0.0722*im[:,:,2]).astype(np.uint8)
    for y in range(len(out)):
        for x in range(len(out)):
            out[y][x] = im[y%im.shape[0]][x%im.shape[1]]            

    return out.astype(np.uint8)

@jit(cache=True, nopython=False , forceobj = True, looplift = True)
def getImageFromFileMosaicToAffination(namein, affination,  gray=True):
    #try:
        #print(namein)
        gc.collect()
        im = imageio.imread(namein)
        shape = im.shape
        return mosaicImage(im, affination), shape
        
        #return mosaicImage(im, affination), shape
    #except Exception:
    #    print(" ERROR LOADING IMAGE MAN!!!! ", namein)

    #out =np.zeros([1, 1, 3], dtype=np.int8)  
    #return out, out.shape   

def checkImageFromFileGetShape(namein):
    try:
        #print(namein)
        im = imageio.imread(namein)
        shape = im.shape
        del im
        gc.collect()
        return shape
        
    except Exception as e:
        print(" ERROR LOADING IMAGE MAN!!!! ", namein)

    return (1, 1, 3) 

def getImageFromFileGetShape(namein, resize, height = 0, gray=False):
    try:
        #print(namein)
        im = imageio.imread(namein)
        shape = im.shape
        im = resizeImage(im, resize, height)
        if gray:
            try:
                im = (0.2126*im[:,:,0]+0.7152*im[:,:,1]+0.0722*im[:,:,2]).astype(np.uint8)

            except:
                pass

        return im.astype(np.uint8), shape
        
    except Exception as e:
        print(" ERROR LOADING IMAGE MAN!!!! ", namein)
        print(" ERROR ", e)

    out =np.zeros([1, 1, 3], dtype=np.int8)  
    return out, out.shape   

#@jit(cache=True, nopython=False)
def getFileNamesFrom(basepath):
    for entry in os.listdir(basepath):
        if os.path.isfile(os.path.join(basepath, entry)):
            yield entry

def getFileNamesFromList(basepath):
    out = []
    print("getFileNamesFromList") 
    print(basepath)
    for entry in os.listdir(basepath):
        if os.path.isfile(os.path.join(basepath, entry)):
            out.append(entry)
    print(len(out))
    return out

def getFileIfExistFrom(basepath, file):
    if os.path.isfile(os.path.join(basepath, file)):
            return file
    return None

def draw_square(shape,proportion = 0.707):
    '''
    Input:
    shape    : tuple (height, width)
    diameter : scalar
    
    Output:
    np.array of shape  that says True within a circle with diamiter =  around center 
    '''
    #assert len(shape) == 2
    TF = np.zeros(shape,dtype=np.bool)
    center = np.array(TF.shape)/2.0
    offset = int((shape[0] * (1 - proportion)) * 0.5)
    for iy in range(int(shape[0] * proportion)):
        for ix in range(int(shape[1] * proportion)):
            TF[iy+offset,ix + offset] = 1
    return(TF)


def draw_rombo(shape,proportion = 0.707):
    '''
    Input:
    shape    : tuple (height, width)
    diameter : scalar
    
    Output:
    np.array of shape  that says True within a circle with diamiter =  around center 
    '''
    #assert len(shape) == 2
    TF = np.zeros(shape,dtype=np.bool)
    center = np.array(TF.shape)/2.0
    offset = int((shape[0] * (1 - proportion)) * 0.5)
    for iy in range(int(shape[0] * proportion)):
        for ix in range(int(shape[1] * proportion)):
            TF[iy+offset,ix + offset] = 1

    TF = ndimage.rotate(TF, 45, reshape=False) #rombo filter
    return(TF)


@jit(cache=True, nopython = True)
def  getImageLuminosities(image):
    return getVectorLuminosities(image.reshape(-1))
    """
    average = 0 
    direct  =0 
    direct_sum = 0
    inverse = 0 
    inverse_sum = 0
    interval = [sys.maxsize,-sys.maxsize-1]
    img_r =  image.reshape(-1)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            average += image[x][y] / (image.shape[0] * image.shape[1])
            direct += image[x][y] * image[x][y]
            direct_sum += image[x][y]
            inverse += 1.0
            inverse_sum += 1.0 / image[x][y]
            if interval[0] > image[x][y]:
               interval[0] = image[x][y]
            
            if interval[1] < image[x][y]:
               interval[1] = image[x][y]
    
    direct /= direct_sum
    inverse /= inverse_sum
    
    return average, direct, inverse, interval 
    """

def  getPonderations(image):
   
    direct  =0 
    direct_sum = 0
    inverse = 0 
    inverse_sum = 0
    interval = [sys.maxsize,-sys.maxsize-1]
    for x in range(len(image)):
            #average += image[x] / len(image)
            direct += image[x] * image[x]
            direct_sum += image[x]
            inverse += 1.0
            inverse_sum += 1.0 / image[x]

    direct /= direct_sum
    inverse /= inverse_sum
    
    return  direct, inverse

@jit(cache=True, nopython = True)
def  getVectorLuminosities(image):
    #mode_info = stats.mode(image) 
    #median = np.median(image)
    #stdv = np.std(image)
    direct  =0 
    direct_sum = 0
    inverse = 0 
    inverse_sum = 0
    #interval = [sys.maxsize,-sys.maxsize-1]
    for x in range(len(image)):
            #average += image[x] / len(image)
            #direct += image[x].astype(np.uint64) * image[x]
            direct += image[x] * image[x] * 0.000001
            direct_sum += image[x] * 0.000001
            if image[x] < 1:
                inverse += image[x] *  1.0 / (1.0 + np.abs(image[x]))
                inverse_sum += 1.0 / (1.0 + np.abs(image[x]))
            
            else:
                inverse += 1
                inverse_sum += 1.0 / image[x]

    if direct_sum != 0:
        direct /= direct_sum
    
    if inverse_sum != 0:
        inverse /= inverse_sum
        #inverse= len(image) / inverse_sum
    
    return  direct, inverse    

def  getVectorLuminosities2(image):
    mode_info = stats.mode(image) 
    median = np.median(image)
    stdv = np.std(image)
    direct  = 0.0
    direct_sum = 0.0
    inverse =  0.0

    inverse2=0
    interval = [sys.maxsize,-sys.maxsize-1]
    for x in range(len(image)):
        #average += image[x] / len(image)
  
        value = image[x]

        direct += value **2
        direct_sum += value

        if interval[0] > value:
            interval[0] = value
            
        if interval[1] < value:
            interval[1] = value
        
        if value == 0:
            continue

        inverse2 +=  1 /value **2
        inverse += 1 /value 
    inverse = inverse / inverse2
    direct /= direct_sum
    
    return mode_info[0][0], median, stdv, direct, inverse, interval, direct_sum  



def spiralFractalDimension(image, limit = 2*np.pi):
    centerx = image.shape[1] / 2
    centery = image.shape[0] / 2
    
    advance_rad1 = 0
    advance_rad2 = 0
    pos1 =  np.complex(centery,  centerx)
    pos2 =  np.complex(centery,  centerx)
    accumulation1 = 0
    accumulation2 = 0
    # centro del espiral, origen...
    #empiezas en el centro, avanzas la etapa computada en función del espacio? no la etapa es  siempre la misma, un pixel.... te mueves con un vector normalizado, haces un floor, empiezas des del pixel, siempre llegas a un pixel te sigues moviendo des de ahi? o sampleas la imagen?
    #ajusta el spiral para que de una vuelta en función del tamaño del cuadrado, es el mismo espiral escalado...
    a_step = 2 / (centerx * centery) #resolución de la grid.... igual tienes que hacerlo con la mitad de la grid ya que te mueves des del centro
    #1 radianes un arco de longitud 1, un pixel tiene longitud uno... entonces tnemosun monton de pixeles, avanzamos segun la densidad de estos pixeles, la mitad del cuadrado grande...
    
    prev_height1 = image[int(np.round(pos1.real))][int(np.round(pos1.imag))].astype(np.float)
    prev_height2 = image[int(np.round(pos2.real))][int(np.round(pos2.imag))].astype(np.float)
    while advance_rad1/limit <=1 and advance_rad2/limit <=1:
        #print("loop " + str(advance_rad1))
        coordpos1 = [int(np.round(pos1.real))-1, int(np.round(pos1.imag))-1]
        if coordpos1[0] <0:
            coordpos1[0] = 0
        elif coordpos1[0] >= image.shape[0]:
            coordpos1[0] = image.shape[0] -1
        if coordpos1[1] <0:
            coordpos1[1] = 0
        elif coordpos1[1] >= image.shape[1]:
            coordpos1[1] = image.shape[1] -1
        #image[coordnpos1[0]][coordnpos1[1]]= 255
        #print("begin the coordinate1")
        #print(coordnpos1)

        coordpos2 = [int(np.round(pos2.real))-1, int(np.round(pos2.imag))-1]
        if coordpos2[0] <0:
            coordpos2[0] = 0
        elif coordpos2[0] >= image.shape[0]:
            coordpos2[0] = image.shape[0] -1
        if coordpos2[1] <0:
            coordpos2[1] = 0
        elif coordpos2[1] >= image.shape[1]:
            coordpos2[1] = image.shape[1] -1
        #image[coordpos2[0]][coordpos2[1]] = 255
        #print("begin the coordinate2")
        #print(coordpos2)

        advance_rad1 += a_step
        advance_rad2 += a_step

        A1 = (0.5 * centerx + 0.5 * centery) * 0.5 * advance_rad1/limit
        A2 = (0.5 * centerx + 0.5 * centery) * 0.5 * advance_rad2/limit

        pulse1 =  np.complex(A1, advance_rad1)
        pulse2 =  np.complex(A2, advance_rad2 + np.pi)

        next_pos1 = np.complex(centery, centerx) + np.exp(pulse1)
        next_pos2 = np.complex(centery, centerx) + np.exp(pulse2)
        #print("positions")
        #print(advance_rad1)
        #
        #print(next_pos1)
        #print("invert position")
        #print(advance_rad2)
        #print(pos2)
        #print(next_pos2)
        #print("")
        #print("iteration "+str(advance_rad1))
        #print(" position1 " + str(int(np.round(pos1.real))-1)+" "+str(int(np.round(pos1.imag))-1))
        #print(pos1)

        #print(" position2 " + str(int(np.round(pos2.real))-1)+" "+str(int(np.round(pos2.imag))-1))
        #print(pos2)
        coordnextpos1 = [int(np.round(next_pos1.real))-1, int(np.round(next_pos1.imag))-1]
        if coordnextpos1[0] <0:
            coordnextpos1[0] = 0
        elif coordnextpos1[0] >= image.shape[0]:
            coordnextpos1[0] = image.shape[0] -1
        if coordnextpos1[1] <0:
            coordnextpos1[1] = 0
        elif coordnextpos1[1] >= image.shape[1]:
            coordnextpos1[1] = image.shape[1] -1
        height1 = image[coordnextpos1[0]][coordnextpos1[1]].astype(np.float)
        #print("the coordinate1")
        #print(coordnextpos1)

        coordnextpos2 = [int(np.round(next_pos2.real))-1, int(np.round(next_pos2.imag))-1]
        if coordnextpos2[0] <0:
            coordnextpos2[0] = 0
        elif coordnextpos2[0] >= image.shape[0]:
            coordnextpos2[0] = image.shape[0] -1
        if coordnextpos2[1] <0:
            coordnextpos2[1] = 0
        elif coordnextpos2[1] >= image.shape[1]:
            coordnextpos2[1] = image.shape[1] -1
        height2 = image[coordnextpos2[0]][coordnextpos2[1]].astype(np.float)

        #print("the coordinate2")
        #print(coordnextpos2)
        
        #image[coordpos1[0]][coordpos1[1]]= 255
        #image[coordpos2[0]][coordpos2[1]]= 0
        pos1 = next_pos1
        pos2 = next_pos2

        difference1 = (height1 - prev_height1) / a_step
        difference2 = (height2 - prev_height2) / a_step

        accumulation1 += difference1 
        accumulation2 += difference2
        
        prev_height1 = height1
        prev_height2 = height2
        
    return accumulation1 + accumulation2



def getNeighbours(position):
    return [position+ np.array([1,0]).astype(np.int), position+ np.array([-1,0]).astype(np.int), position+ np.array([0,1]).astype(np.int), position+ np.array([0,-1]).astype(np.int)]

def bfsFractalDimension(image, limit = 2*np.pi): #mira el heat equation, calcula el gradiente en el pixel... no del path o salto....
    centerx = image.shape[1] // 2
    centery = image.shape[0] // 2
    previous = np.array([centery, centerx]).astype(np.int)
    queue = [previous]
    map = {}
    out = float(0)
    while len(queue) > 0:
        point_pop = queue.pop(0)
        neighbours = getNeighbours(point_pop)
        for i in range(len(neighbours)):
            point = neighbours[i]
            if not (point[0]>=0 and point[1]>=0):
                continue
            
            if not (point[0]< image.shape[0] and point[1]<image.shape[1]):
                continue
            
            visited = None
            if point[0] not in map:
                map[point[0]]  = set()
            
            if point[1] in map[point[0]]:
                    continue

            map[point[0]].add(point[1])
            queue.append(neighbours[i])
        
        distance = np.linalg.norm(previous - point_pop)
        if distance == 0:
            continue

        #compute the gradient of every point, not the gradient like you traveling as some weird jumps can appear...
        #compute te instant frequency jake
        out += (int(image[point_pop[0]][point_pop[1]]) - int(image[previous[0]][previous[1]])) / distance
        previous = point_pop
        #do whatever with point
    return out
                
        

        
       

def sampler(pos, image, ponderation): #pass an sice of 2 * 2 for exe
    pass #you receive a position, floor, ceil, and with the difference floating point between floor and ceil make a poderation... sample with linear, bilinear, or whatever... 
    #return getVectorLuminosities2()



def sphericals( point):
    r = np.linalg.norm(point)
    theta = 0
    phi = 0
    if r != 0:
        if point[0] != 0:
            theta = np.arctan(point[1] / point[0])

        else:
            theta = math.pi / 2
          
        phi = np.arccos(point[2] / r)
    
    return np.array([phi, theta, r]);

def radiansToCartesians(radians, blue = 255, amplitude = 255, absolute = True):
    factor_normalization  = 17
    radians = radians / (np.pi)
    value = np.round(factor_normalization * radians)
    radians = np.pi * value / factor_normalization  

    red = 51 * radians / (np.pi) 
    if red > 255:
        red = 255

    s = np.sin(radians)
    c = np.cos(radians)
    return np.array([red, 255 * np.abs(3*radians)**2, 255 * np.abs(3*radians)**2]).astype(np.uint8)  

def radiansToCartesians2(radians, blue = 255, amplitude = 255, absolute = True):
    factor_normalization  = 51
    blue = blue / amplitude
    value = np.round(factor_normalization*blue)
    blue = 255 * value / factor_normalization

    radians = radians / np.pi
    value = np.round(24 * radians)
    radians = np.pi * value / 24 

    s = np.sin(radians)
    c = np.cos(radians)
    return np.array([amplitude * np.abs(c), amplitude * np.abs(s), blue]).astype(np.uint8)  


def cartesians( point):
    if len(point) < 3:
        return
    
    x = point[2] * np.sin(point[0]) * np.cos(point[1])
    y = point[2] * np.sin(point[0]) * np.sin(point[1])
    z = point[2] * np.cos(point[0])
    return np.array([x, y, z]); 


def save_histogram(image_name, savepath, queue):
    img = getImageFromFile(image_name, 500)#Image.open(image_name)
    h, w, comp3 = img.shape
    t=0
    print(h, " ", w)
    print(len(img[::]), " ", len(img[0][::]))
    
    output = np.zeros([91, 91, 3], np.float64)
    ponderation = np.ones([91, 91, 3], np.float64)
    for position in product(range(h), range(w)):
        sph = sphericals(img[position])
        degrees = np.array([math.degrees(sph[0]), math.degrees(sph[1])])
        pos = (int(round(degrees[0])), int(round(degrees[1]))) 
        output[pos] += img[position] * sph[2]
        ponderation[pos] += sph[2]
        
    output /= ponderation
    print(output)   
    imageio.imwrite(uri=savepath, im=output.astype(np.uint8))  

def draw_cicle(shape,diamiter):
    '''
    Input:
    shape    : tuple (height, width)
    diameter : scalar
    
    Output:
    np.array of shape  that says True within a circle with diamiter =  around center 
    '''
    assert len(shape) == 2
    TF = np.zeros(shape,dtype=np.bool)
    center = np.array(TF.shape)/2.0

    for iy in range(shape[0]):
        for ix in range(shape[1]):
            TF[iy,ix] = (iy- center[0])**2 + (ix - center[1])**2 < diamiter **2
    return(TF)

def draw_square(shape,proportion = 0.707):
    '''
    Input:
    shape    : tuple (height, width)
    diameter : scalar
    
    Output:
    np.array of shape  that says True within a circle with diamiter =  around center 
    '''
    assert len(shape) == 2
    TF = np.zeros(shape,dtype=np.bool)
    center = np.array(TF.shape)/2.0
    offset = int((shape[0] * (1 - proportion)) * 0.5)
    for iy in range(int(shape[0] * proportion)):
        for ix in range(int(shape[1] * proportion)):
            TF[iy+offset,ix + offset] = 1
    return(TF)

def draw_gaussian(gaussian_factor, shape):
    '''
    Input:
    shape    : tuple (height, width)
    diameter : scalar
    
    Output:
    np.array of shape  that says True within a circle with diamiter =  around center 
    '''
    assert len(shape) == 2
    TF = np.zeros(shape,dtype=np.bool)
    center = np.array(TF.shape)/2.0
    for iy in range(int(shape[0] )):
        for ix in range(int(shape[1] )):
            y = iy - 0.5 * shape[0]
            x = ix - 0.5 * shape[1]
            factory = gaussian_factor * shape[0]
            factorx = gaussian_factor * shape[1]
            TF[iy][ix] =  1/( 1 + x**2/(factorx**2)+y**2/(factory**2) )
            
    return(TF)

def draw_rombo(shape,proportion = 0.707):
    '''
    Input:
    shape    : tuple (height, width)
    diameter : scalar
    
    Output:
    np.array of shape  that says True within a circle with diamiter =  around center 
    '''
    #assert len(shape) == 2
    TF = np.zeros(shape,dtype=np.bool)
    center = np.array(TF.shape)/2.0
    offset = int((shape[0] * (1 - proportion)) * 0.5)
    for iy in range(int(shape[0] * proportion)):
        for ix in range(int(shape[1] * proportion)):
            TF[iy+offset,ix + offset] = 1

    TF = ndimage.rotate(TF, 45, reshape=False) #rombo filter
    return(TF)

def make_figure(fft, img, title):
    fig, (ax) = plt.subplots(2,2)

    fig.suptitle('Fourier Footprinting', fontsize=20)

    ax[0,0].imshow(fft_img_values(np.abs(img)) , cmap="gray")
    ax[0,0].set_title('logarithmic')

    ax[1,0].imshow(np.abs(img) , cmap="gray")
    ax[1,0].set_title('absolute')

    ax[0,1].imshow(img.real , cmap="gray")
    ax[0,1].set_title('real')

    ax[1,1].imshow(ifft_img_phase(np.angle(fft)) , cmap="gray")
    ax[1,1].set_title(title)

def make_figure_3d(fft, title):
    fig = plt.figure()
    fig.suptitle(title, fontsize=20)
    ax = fig.add_subplot(111, projection='3d')

    xx, yy = np.mgrid[0:fft.shape[0], 0:fft.shape[1]]


    
def add_to_figure_3d(fig, subplot, fft, title):
    ax = fig.add_subplot(subplot, projection='3d')

    xx, yy = np.mgrid[0:fft.shape[0], 0:fft.shape[1]]

    # create the figure
    #fig = plt.figure()
    ax.plot_surface(xx, yy, fft ,rstride=1, cstride=1, cmap=plt.cm.jet,
                linewidth=0)
    ax.set_title(title)


def add_to_figure_image(fig, subplot, img, title, vmin=0, vmax=0, interpolation = 'nearest'):
    ax = fig.add_subplot(subplot)
    if vmin != vmax:
        ax.imshow(img , cmap="gray", vmin=vmin, vmax=vmax, interpolation=interpolation)
    
    else:
        ax.imshow(img , cmap="gray")

    ax.set_title(title) 


def imshow_fft(absfft):
    magnitude_spectrum = 20*np.log(absfft + 0.000001)
    return (plt.imshow(magnitude_spectrum,cmap="gray"))

def imshow_ifft(absfft):
    magnitude_spectrum = 20*np.log(absfft + 0.000001)
    print(absfft.shape)
    return(plt.imshow(magnitude_spectrum, cmap="gray"))

def ifft_img(absfft):
    #print("ifft img")
    magnitude_spectrum = np.log(absfft + 0.000001)
    #print(absfft.shape)
    return magnitude_spectrum

def fft_img_values(absfft):
    #print("fft img values")
    magnitude_spectrum = np.log(absfft + 0.000001)
    #print(absfft.shape)
    return magnitude_spectrum




def ifft_img_phase(absfft):
    #print(np.rad2deg(absfft[400:600,400:600]) +180)
    magnitude_spectrum = ((np.rad2deg(absfft) + 180)* 255/360).astype(np.uint8)# 20*np.log(absfft + 5000 + 0.000001)
    #print(magnitude_spectrum)
    return magnitude_spectrum

#[(fourier_image, titul, subplot, 3dbool), ...]
class InputPFF:
    def __init__(self, fourier_image,title, vmin=0, vmax=0, interpolation='nearest', plot_3d=False):
        #print("init")
        #print(fourier_image.shape)
        self.fourier_image = fourier_image
        self.title = title 
        self.plot_3d = plot_3d
        self.vmin = vmin
        self.vmax = vmax
        self.interpolation = interpolation

@jit(cache=True, nopython=False, forceobj=True, looplift=True)
def fourierTransform(img):
    #print("fourier Transform ", img.shape)
    fft = np.zeros_like(img ,dtype=complex)
    #print(img.shape)
    ichannels =0
    if len(fft.shape) >2:
        ichannels = fft.shape[2] - 1

    for ichannel in range(ichannels+1):
        if ichannels:
            fft[:,:,ichannel] = np.fft.fftshift(np.fft.fft2(img[:,:,ichannel]))
            continue
        fft = np.fft.fftshift(np.fft.fft2(img))
    return fft

@jit(cache=True)
def inv_FFT_all_channel(fft_img, trans =True):
        ichannels =0
        if len(fft_img.shape) >2:
            ichannels = fft_img.shape[2] - 1

        img_reco = []
        for ichannel in range(ichannels + 1):
            if ichannels:
                img_reco.append(np.fft.ifft2(np.fft.ifftshift(fft_img[:,:,ichannel])))
            
            else:
                img_reco.append(np.fft.ifft2(np.fft.ifftshift(fft_img[:,:])))

        img_reco = np.array(img_reco)
        #try:
        if trans:
            img_reco = np.transpose(img_reco,(1,2,0))
        
        else:
            img_reco = np.transpose(img_reco) 
               
        #except:
        #    print("transpose exception...")
        #    return img_reco[0]
    
        return img_reco 


def plot_fourier_figures(inputPFF_list, name = "fig"): #plot recovered image, modul image, angle image, 3d image in another figure.
    fig = plt.figure()
    subplot = len(inputPFF_list);
    matrix = str(subplot)+"3"
    #print("matrix " + matrix)
    index=1
    for i in range(len(inputPFF_list)):
        input_pff = inputPFF_list[i]
        #print(input_pff.fourier_image.shape)
        img_reco = inv_FFT_all_channel(input_pff.fourier_image, 0)
        #print("image reco ")
        #print(img_reco.shape)
        #print("what")
        subplot = matrix+str(index)
        add_to_figure_image(fig, int(subplot), fft_img_values(np.abs(input_pff.fourier_image)), input_pff.title + " absolute")
        index+=1
        subplot = matrix+str(index)
        add_to_figure_image(fig, int(subplot), ifft_img_phase(np.angle(input_pff.fourier_image)), input_pff.title + " angle")
        index+=1
        subplot = matrix+str(index)
        #print("hey you")
        add_to_figure_image(fig, int(subplot), np.abs(img_reco), input_pff.title + " recovered")
        index+=1
        if input_pff.plot_3d:
            fig = plt.figure()
            add_to_figure_3d(fig, 121, ifft_img_phase(np.angle(input_pff.fourier_image)), input_pff.title + " angle")
            add_to_figure_3d(fig, 122, fft_img_values(np.abs(input_pff.fourier_image)), input_pff.title + " absolute")
    fig.savefig(name+"_fourier.png");
    fig.clear()
    plt.close(fig)

def plot_figures(inputPFF_list, name="fig.png", save= False): #plot recovered image, modul image, angle image, 3d image in another figure.
    fig = plt.figure()
    #print(len(inputPFF_list) )
    subplot = int(np.ceil(len(inputPFF_list) / 3));
    matrix = str(subplot)+"3"
    if len(inputPFF_list) <=3:
        matrix = "1"+str(len(inputPFF_list))
    
    #print("matrix " + matrix)
    index=1
    for input_pff in inputPFF_list:
            subplot = matrix+str(index)
            print(subplot)
            add_to_figure_image(fig, int(subplot), input_pff.fourier_image, input_pff.title, input_pff.vmin, input_pff.vmax, input_pff.interpolation)
            index+=1

    print("figure " + name+".png")
    if save:
        fig.savefig(name+".png");
        fig.clear()
        plt.close(fig)
    
    else:
        plt.show()
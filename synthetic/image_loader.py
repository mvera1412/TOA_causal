import numpy as np
from skimage.color import rgb2gray, rgba2rgb
from skimage import io
import cv2 as cv

def imretina(pathfile):
    #pathfile='images/'
    Nimage=20 # Número de imagenes para segmentar maximo=20
    px=64 # cantidad de puntos lado de imagen cuadrada
    xo=128
    yo=128 # posición inicial de donde comienza la segmentación
    nf=6 # Cantidad de segmentos por eje, total es nf*nf
    MI=np.zeros((px*px,Nimage*nf*nf)) # Inicializo matriz de imágenes
    cont1=0; # Contador del segmento
    for i1 in range(21,21+Nimage):
        tifile=pathfile+str(i1)+'_training.tif'
        #tifile=pathfile+str(i1)+'_manual1.tif'
        imrgb=io.imread(tifile) # leo la imagen en color
        grayim=rgb2gray(imrgb) # paso a escala de grises
        #grayim=imrgb # Las manual ya vienen en escala de grises
        for k in range(1,nf+1):
            xi=xo+px*(k-1);
            xf=xi+px;
            for kk in range(1,nf+1):
                yi=yo+px*(kk-1);
                yf=yi+px;
                aux=grayim[xi:xf,yi:yf]
                aux=cv.GaussianBlur(aux,(5,5),0) # Filtro para suavizar la imagen
                aux=aux.ravel(); # 1 segmento de la imagen original
                aux=aux/np.max(aux)
                aux=aux*-1+1;
                aux=aux/np.max(aux)
                MI[:,cont1]=aux;
                cont1=cont1+1; # Aumento contador de segmento

    return(MI)

def imletnum(pathfile):
    #pathfile='letnum/'
    Nimage = 40 # Número de imagenes maximo=40
    px = 64 # cantidad de puntos lado de imagen cuadrada
    MI = np.zeros((px*px,Nimage)) # Inicializo matriz de imágenes
    for i1 in range(1,Nimage+1):
        tifile = pathfile+'let'+str(i1)+'.tif'
        imrgb = io.imread(tifile) # leo la imagen en color
        #grayim = rgb2gray(imrgb) # paso a escala de grises
        grayim= rgb2gray(rgba2rgb(imrgb))
        img = grayim.ravel()
        img = img/np.max(img)
        img = img*-1 + 1
        MI[:,i1-1] = img
    return(MI)

def paqim(pathfile):
    #pathfile='images/'
    Nimage=20 # Número de imagenes para segmentar maximo=20
    px=128 # cantidad de puntos lado de imagen cuadrada
    xo=128
    yo=128 # posición inicial de donde comienza la segmentación
    nf=3 # Cantidad de segmentos por eje, total es nf*nf
    MI=np.zeros((px*px,Nimage*nf*nf)) # Inicializo matriz de imágenes
    cont1=0; # Contador del segmento
    for i1 in range(21,21+Nimage):
        tifile=pathfile+str(i1)+'_training.tif'
        #tifile=pathfile+str(i1)+'_manual1.tif'
        imrgb=io.imread(tifile) # leo la imagen en color
        grayim=rgb2gray(imrgb) # paso a escala de grises
        #grayim=imrgb # Las manual ya vienen en escala de grises
        for k in range(1,nf+1):
            xi=xo+px*(k-1);
            xf=xi+px;
            for kk in range(1,nf+1):
                yi=yo+px*(kk-1);
                yf=yi+px;
                aux=grayim[xi:xf,yi:yf]
                aux=cv.GaussianBlur(aux,(5,5),0) # Filtro para suavizar la imagen
                aux=aux.ravel(); # 1 segmento de la imagen original
                aux=aux/np.max(aux)
                aux=aux*-1+1;
                aux=aux/np.max(aux)
                MI[:,cont1]=aux;
                cont1=cont1+1; # Aumento contador de segmento

    return(MI)
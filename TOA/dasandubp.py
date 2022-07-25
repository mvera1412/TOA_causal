# Modules:
import numpy as np
import numpy.matlib
from scipy.interpolate import interp1d
from scipy.fftpack import fft, ifft

###############################################################################
def ind2sub(array_shape, ind):
    # Gives repeated indices, replicates matlabs ind2sub
    rows = (ind.astype("int64") // array_shape[1])
    cols = (ind.astype("int64") % array_shape[1])
    return (rows, cols)

###############################################################################
def SensorMaskCartCircleArc(circle_radius, circle_arc, num_sensor_points):
    th = np.linspace(0, circle_arc * np.pi / 180, num_sensor_points + 1)
    th = th[0:len(th) - 1]  # Angles
    posSens = np.array([circle_radius * np.cos(th), circle_radius * np.sin(th), np.zeros((len(th)))])  # position of the center of the sensors
    return posSens # (3,Ns)

###############################################################################
def DAS(nx,dx,dsa,posSens,vs,t,p):
    """
    Traditional Reconstruction Method "Delay and Sum" for 2-D OAT
    The output P0 is the initial pressure [P0] = (N,) where N is the total pixels
    of the image region.
    
    nx: number of pixels in the x direction for a 2-D image region
    dx: pixel size  in the x direction [m]
    dsa: distance sensor array [m]
    posSens: position of the center of the detectors (3,Ns) [m]
    vs: speed of sound (homogeneous medium) [m/s]
    t: time samples (Nt,) [s]
    p: OA measurements (Ns,Nt) where Ns: number of detectors [Pa]
    
    References:
        [1] X. Ma, et.al. "Multiple Delay and Sum with Enveloping Beamforming 
        Algorithm for Photoacoustic Imaging",IEEE Trans. on Medical Imaging (2019).
    """  
    
    # GET Ns, Nt and N
    Ns=p.shape[0]
    Nt=p.shape[1]
    N = nx**2; # Total pixels 
    
    # 2-D IMAGE REGION GRID
    ny = nx  # nx = ny. Rectangular grid
    N = nx * ny  
    dy = dx  # pixel size in the y direction 

    originX = np.ceil(nx / 2) # Set image region origin in the x direction
    originY = np.ceil(ny / 2) # Set image region origin in the y direction
    y, x = ind2sub([nx, ny], np.linspace(0, N - 1,N)) # nornalized coordinate pixel position [x]=(N,); [y]=(N,) 
    rj=np.array([(x-originX)*dx,(y-originY)*dy]) # pixel position [rj]=(2,N)
    rj=np.transpose(rj) # (N,2)
    Rj=np.reshape(rj,(1,N*2)) # (1,2*N)
    Rj=np.repeat(Rj,Ns,axis=0) # (Ns,2*N)
    Rj=np.reshape(Rj,(Ns*N*2,1)) # (2*N*Ns,1)
        
    # DETECTOR POSITIONS
    rs=posSens[0:2,:] # (2,Ns)
    rs=np.transpose(rs) # (Ns,2)
    Rs=np.repeat(rs,N,axis=0) # (N*Ns,2)
    Rs=np.reshape(Rs,(Ns*N*2,1)) # (2*N*Ns,1)
 
    # TIME GRID
    Tau=(Rs-Rj)/vs
    Tau=np.reshape(Tau,(Ns*N,2))
    Tau=np.linalg.norm(Tau,ord=2,axis=1) # Get norm 2 by row
    Tau=np.reshape(Tau,(Ns,N))
    
    # OBTAIN 2-D IMAGE
    P0=np.zeros((N,))
    t = np.reshape(t,(1,Nt))
    for i in range(0,Ns):
        fp=interp1d(t[0,:],p[i,:]) #interpolación lineal de la mediciones para los tiempo GRILLA
        aux=fp(Tau[i,:])
        P0=P0+aux
    
    #P0=np.reshape(P0,(nx,nx)) 
    return P0 

###############################################################################
def UBP(nx,dx,dsa,posSens,vs,t,pmed):
    """
    Traditional Reconstruction Method "Universal Back-Projection" for 2-D OAT
    It is assumed detector are on a circunference surrounding the image region,
    all in the XY plane. It is assumed that the detectors are in the same plane
    as the image and they are distributed in a circunference of radius dsa.     
    
    nx: number of pixels in the x direction for a 2-D image region
    dx: pixel size  in the x direction [m]
    dsa: distance sensor array [m]
    posSens: position of the center of the detectors (3,Ns) [m]
    vs: speed of sound (homogeneous medium) [m/s]
    t: time samples (Nt,) [s]
    pmed: OA measurements (Ns,Nt) where Ns: number of detectors [Pa]
    
    Reference:
        [1] M. Xu, et al., "Universal back-projection #algorithm for 
        photoacoustic computed tomography", Phys. Rev. E 71, p. 016706 (2005).
    """    
    
    # GET Ns, Nt and N
    Ns=pmed.shape[0]
    #Nt=pmed.shape[1]
    
    # 2-D IMAGE REGION GRID
    ny = nx  # nx = ny. Rectangular grid
    #N = nx * ny  # Total pixels 
    dy = dx  # pixel size in the y direction 

    originX = np.ceil(nx / 2) # Set image region origin in the x direction
    originY = np.ceil(ny / 2) # Set image region origin in the y direction
    xf = (np.arange(0,nx)-originX)*dx # (nx,)
    yf = (np.arange(0,ny)-originY)*dy # (ny,)
    Yf, Xf = np.meshgrid(yf, xf)
        
    # DETECTOR POSITIONS
    rs=posSens[0:2,:] # (2,Ns)
    xd = rs[0,:]; xd = np.reshape(xd,(Ns,))
    yd = rs[1,:]; yd = np.reshape(yd,(Ns,))
    
    # BACKPROJECTION
    fs = 1/(t[1]-t[0])  #sampl freq
    NFFT = 2048
    fv = fs/2*np.linspace(0,1, int(NFFT/2+1))
    fv2 = -np.flipud(fv)  #for
    fv2 = np.delete(fv2,0)
    fv2 = np.delete(fv2,-1)
    fv3 = np.concatenate((fv,fv2),0)  #ramp filter for positive and negative freq components
    k = 2*np.pi*fv3/vs #wave vector

    ds = 1e-3*20e-3  #active area of sensor (assuming the size of a line detector)
    pf = np.empty([len(k)])
    pnum = 0
    pden = 0
    
    cont = -1
    for i1 in range(0,Ns):
        cont = cont + 1
        X2 = (Xf-rs[0,cont])**2
        Y2 = (Yf-rs[1,cont])**2
        dist = np.sqrt(X2+Y2)  #distance from detector to each pixel in imaging field
        distind = np.round(dist*(fs/vs)) #convert distance to index
        distind = distind.astype(int) #convert to integer for indexing
        p = pmed[cont,:]
        pf = ifft(-1j*k*fft(p,NFFT)) #apply ramp filter
        b = 2*p - 2*t*vs*pf[0:len(p)]
        b1 = b[distind]
        # Calculation dOmega assuming detectors on a circunference surrounding the sample
        rsni = np.sqrt(rs[0,cont]**2+rs[1,cont]**2)
        etta = -1*(rs[0,cont]*(Xf-rs[0,cont]) + rs[1,cont]*(Yf-rs[1,cont]))/rsni
        omega = (ds/dist**2)*etta/dist
        pnum = pnum + omega*b1
        pden = pden + omega

    pg = pnum/pden
    pgmax = pg[np.nonzero(np.abs(pg) == np.amax(abs(pg)))]   #np.amax(complex array) will return the element with maximum real value, not maximum absolute value as in Matlab
    pfnorm = np.real(pg/pgmax)
    return pfnorm

###############################################################################
def bpdlv1(t,PVm,vs,Dg,TC,ArcBar,tipo,rotimag,mimag):
    """
    BPDLv1 Esta función realiza la reconstrucción de una imagen a partir de la
      medición de la integral de la presión generada por el efecto optoacústico Po. 
      La misma se realiza sobre una curva C donde se disponen Ns sensores. Se
      supone una velocidad del sonido cs uniforme y constante. 
      El algoritmo está basado en el trabajo de:
      P. Burgholzer, et.al. 'Temporal back-propagation algoritms for 
      photoacoustic tomography with integrating line detectors', Inverse
      Problems 23 (2007) S65-S80.

      La sintaxis completa es la siguiente:

         x,y,Po=bpdlv1(t,PVm,vs,Dg,TC,ArcBar,tipo,rotimag,mimag)

      Acontinuacion se detallan los parametros de entrada: 
      t[m,1]: eje de tiempo en [s].
      PVm[m,k]: matriz con la presión medidas en función de t(m,1) por los sensore en la posición k.
      vs: velocidad del sonido en el medio en [m/s].
      Dg: tamaño de un elemento de la grilla de la imagen en [m].
      TC: tamaño característico de la superficie de detección en [m].
      ArcBar: arco barrido por el sensor [grados].
      tipo: elección del método: 1 UBP; 2 FF; 3 MFF, 4 SYM, 5 ASYM.
      rotimag: circularshift para rotar la imagen, debe ser menor que k
      mimag: espejar imagen 'n' no espejar 'h' horizontal, 'v' vertical y 'b' vertical y horizontal.

    Ejemplo:
           x,y,Po=bpdlv1(t,PVm,1485,10e-6,12.512e-3,348,1,30,'n');
 
    """

    # Definición de constantes
    Os=4*np.pi; # 2*pi para superficie de deteccion plana y 4*pi para superficies esfericas o cilindricas

    # Rotacion de la imagen
    PVm=np.roll(PVm,rotimag,1)

    # Extracción de datos de la matriz de presiones medidas 
    Nt=np.size(PVm,0); # Cantidad de muestras temporales.
    Ns=np.size(PVm,1); # Cantidad de sensores sobre la curva C
    Dt=t[1,0]-t[0,0]; # Paso de tiempo en [s]
 
    # Angulos donde son dispuestos los sensores
    # Distribución uniforme 
    th = np.linspace(0,ArcBar*np.pi/180,Ns+1); th = th[0:len(th)-1];
    
    x=TC*np.cos(th);
    y=TC*np.sin(th);
    Us = np.zeros((Ns,2))
    Us[:,0]=x;  Us[:,1]=y;

    #Distancia entre sensores sobre la curva C suponiendo distribucion uniforme
    Dttita=ArcBar*np.pi/180/Ns;
    Ds=Dttita*TC;

    # Armado de la grilla de la imagen a reconstruir
    Ng=int(TC/Dg); # Cantidad de elementos de la grilla
    x=np.linspace(-TC,TC,Ng);
    y=np.linspace(-TC,TC,Ng);
    Xg,Yg=np.meshgrid(x,y);

    # Mascara
    RR=np.sqrt(Xg**2+Yg**2);
    Mascara=RR<TC; Mascara=Mascara*1; 
    del RR
    
    # Normalizacion
    t=t*vs;  # Suponiendo cs constante y uniforme, paso el eje de tiempo a distancias
    t=t/(2*TC); # Normaliza respecto del diámetro de la curva C

    Dt=Dt*vs;  
    Dt=Dt/(2*TC);

    Ds=Ds/(2*TC);

    Xg=Xg/(2*TC);
    Yg=Yg/(2*TC);

    Us=Us/(2*TC);

    # Inicialización variables
    a=np.zeros((Nt,Nt));
    b=np.zeros((Nt,Nt));
    q=np.zeros((Nt,Ns));
    qh=np.zeros((Nt,Ns));
    Po=np.zeros((Ng,Ng));

    # Determinación de los coeficientes a[m,mp] y b[m,mp]
    #t=np.reshape(t,(Nt,1))
    MTm=np.reshape(t[0:-1,0],(len(t[0:-1,0]),1))@np.ones((1,Nt-1)); 
    MTm=np.triu(MTm,0)+np.tril(np.ones((Nt-1,Nt-1)),-1);
    MTmp=np.ones((Nt-1,1))@np.reshape(t[0:-1,0],(1,len(t[0:-1,0]))); MTmp=np.triu(MTmp,0)+np.tril(np.ones((Nt-1,Nt-1)),-1);
    MTmp1=np.ones((Nt-1,1))@np.reshape(t[1:,0],(1,len(t[1:,0]))); MTmp1=np.triu(MTmp1,0)+np.tril(np.ones((Nt-1,Nt-1)),-1);
      
    a[0:-1,0:-1]=np.log10(MTmp1+np.sqrt(MTmp1**2-MTm**2))-np.log10(MTmp+np.sqrt(MTmp**2-MTm**2));
    a[0,0]=0;

    del MTm, MTmp, MTmp1  # Liberando espacio

    MTm=np.reshape(t[0:-1,0],(len(t[0:-1,0]),1))@np.ones((1,Nt-1)); MTm=np.triu(MTm,0);
    MTmp=np.ones((Nt-1,1))@np.reshape(t[0:-1,0],(1,len(t[0:-1,0]))); MTmp=np.triu(MTmp,0);
    MTmp1=np.ones((Nt-1,1))@np.reshape(t[1:,0],(1,len(t[1:,0]))); MTmp1=np.triu(MTmp1,0);

    b[0:-1,0:-1]=(np.sqrt(MTmp1**2-MTm**2)-np.sqrt(MTmp**2-MTm**2))-MTmp*a[0:-1,0:-1];

    del MTm, MTmp, MTmp1  # Liberando espacio

    # Determinación de q[m,k]
    MTp=np.reshape(t[3:,0],(len(t[3:,0]),1))@np.ones((1,Ns));
    MT=np.reshape(t[2:-1,0],(len(t[2:-1,0]),1))@np.ones((1,Ns));
    MTr=np.reshape(t[1:-2,0],(len(t[1:-2,0]),1))@np.ones((1,Ns));
    Ro=np.ones((Nt-3,1))@np.reshape(np.sqrt((Us[:,0]**2)+(Us[:,1]**2)),(1,len(np.sqrt((Us[:,0]**2)+(Us[:,1]**2)))));

    if tipo==1: # UBP (Universal BackProjection)
        q[2:-1,:]=(PVm[3:,:]/MTp-PVm[1:-2,:]/MTr)/(2*Dt);
    elif tipo==2:  #FF (Far-Field inversion)
        q[2:-1,:]=(PVm[3:,:]-PVm[1:-2,:])/MT/(2*Dt);
    elif tipo ==3: # MFF (Modified Far-Field inversion)
        q[2:-1,:]=(PVm[3:,:]-PVm[1:-2,:])/MT/(2*Dt);
    elif tipo==4: # SYM (finch, SYMmetric)
        q[2:-1,:]=(PVm[3:,:]*MTp-PVm[1:-2,:]*MTr)/Ro/(2*Dt);
    elif tipo==5: # ASYM (finch, ASYMmetric)
        q[2:-1,:]=MT*(PVm[3:,:]-PVm[1:-2,:])/Ro/(2*Dt);
    else:
        print("Opción no válida, se usa la opción por defecto: UBP")
        q[2:-1,:]=(PVm[3:,:]/MTp-PVm[1:-2,:]/MTr)/(2*Dt);
        
    del MTp, MT, MTr  # Liberando espacio

    # Determinación de qh[m,k]
    qh[0:-1,:]=a[0:-1,0:-1]@q[0:-1,:]+(b[0:-1,0:-1]@(q[1:,:]-q[0:-1,:]))/(Dt);


    # Determinacion dxs/de y dys/de de la curva C
    dx=np.zeros((Ns,1));
    dy=np.zeros((Ns,1));
    dx[0,0]=(Us[1,0]-Us[Ns-1,0])/(2*Ds);
    dx[-1,0]=(Us[0,0]-Us[-2,0])/(2*Ds);
    dy[0,0]=(Us[1,1]-Us[-1,1])/(2*Ds);
    dy[-1,0]=(Us[0,1]-Us[-2,1])/(2*Ds);
    dx[1:-1,0]=(Us[2:,0]-Us[0:-2,0])/(2*Ds);
    dy[1:-1,0]=(Us[2:,1]-Us[0:-2,1])/(2*Ds);

    # Reconstrucción de la imagen Po suponiendo una curva C circular
    for k in range(0,Ns):
        xs=Us[k,0]*np.ones((Ng,Ng)); ys=Us[k,1]*np.ones((Ng,Ng));
        rho=np.sqrt((Xg-xs)**2+(Yg-ys)**2);
        nCx=xs/np.sqrt(xs**2+ys**2); # normal de la curva C en en punto k (se define positiva hacia fuera)
        nCy=ys/np.sqrt(xs**2+ys**2);
        gama=(nCx*(Xg-xs)+nCy*(Yg-ys))/rho;
        
        if tipo==1: # UBP (Universal BackProjection)
            w=rho*np.cos(gama)        
        elif tipo==2:  #FF (Far-Field inversion)
            w=rho*np.cos(gama) 
        elif tipo ==3: # MFF (Modified Far-Field inversion)
            w=rho
        elif tipo==4: # SYM (finch, SYMmetric)
            w=1*np.ones((Ng,Ng));
        elif tipo==5: # ASYM (finch, ASYMmetric)
            w=1*np.ones((Ng,Ng));
        else:
            print("Opción no válida, se usa la opción por defecto: UBP")
            w=rho*np.cos(gama) 
        
        Hc=np.sqrt(dx[k,0]**2+dy[k,0]**2)*Ds;
        qhval=interp1d(np.reshape(t,(len(t),)),qh[:,k]); # Interpolado lineal
        Po=Po-4/Os*qhval(rho)*w*Hc*Mascara; 
    
    # Reflejo
    imob = Po
    # Flip imagen
    if mimag == 'h':
        imob = np.flip(imob,1) # flip horizontal
    if mimag == 'v':
        imob = np.flip(imob,0) # flip vertical
    if mimag == 'b':
        imob = np.flip(imob,0) # flip en ambas dimensiones`
        imob = np.flip(imob,1)
    Po = imob
    
    return(x,y,Po)

###############################################################################

# Modules:
import numpy as np
from numpy import matlib
from scipy import sparse
from scipy.sparse import csc_matrix # Column sparse 
from tqdm import tqdm # progress bar for loops
from scipy.linalg import convolution_matrix
#from scipy.linalg import toeplitz

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
def build_matrix(nx,dx,Ns,posSens,ls,nls,vs,tt,normA,thresh,rsnoise,tdo,tlp=0):
    """
    Model-based Matrix by Spatial Impulse Response Approach -> A: (Ns*Nt,N)
    if tdo == False
        VP = A@P0 # where VP: velocity potencial (Ns*Nt,); P0: initial pressure (N,)
    else:
        P = A@P0 # where P: acoustic pressure (Ns*Nt,)
    
    nx: number of pixels in the x direction for a 2-D image region
    dx: pixel size  in the x direction [m]
    Ns: number of detectors
    posSens: position of the center of the detectors (3,Ns) [m]
    ls: length of the integrating line detector [m]
    nls: number of elements of the divided sensor (discretization)
    vs: speed of sound (homogeneous medium) [m/s]
    tt: time samples (Nt,) [s]
    normA: normalize A? True or False
    thresh: threshold the matrix to remove small entries and make it more sparse 10**(-thresh)
    rsnoise: reduce shot noise and add laser pulse duration effect? True or False
    tdo: apply time derivative operator? True or False
    tlp: laser pulse duration [s], by default is set to zero
    
    Example:
        posSens = SensorMaskCartCircleArc(8e-3, 348, 64)
        tt = np.linspace(0,10e-6,512);
        A = build_matrix(64,125e-6,64,posSens,20e-3,round(20e-3/125e-6*2),1485,tt,True,0,True)
    
    References:
        [1] G. Paltauf, et al., "Modeling PA imaging with scanning focused detector using
        Monte Carlo simulation of energy deposition", J. Bio. Opt. 23, p. 121607 (2018).
        [2] G. Paltauf, et al., "Iterative reconstruction algorithm for OA imaging",
        J. Acoust. Soc. Am. 112, p. 1536 (2002).
    """    
    # Important constants
    Betta = 207e-6  # Thermal expansion coefficient for water at 20 °C [1/K].
    Calp = 4184     # Specific heat capacity at constant pressure for water at 20 °C [J/K Kg].
    rho = 1000      # Density for water or soft tissue [kg/m^3].
    #h0 = 1e4        # h0=mua*FLa; Energy absorbed in the sample per unit volumen (typical value) [J/m^3].
    #p0=100;         # Initial pressure (typical value) [Pa].
    #phi0 = 5e-9     # phi0=-Dt*p0/rho; Initial velocity potential assuming p0=100Pa and Dt=50ns [m^2/s].

    # 2-D IMAGE REGION GRID
    ny = nx  # nx = ny. Rectangular grid
    N = nx * ny  # Total pixels 
    dy = dx  # pixel size in the y direction
    dz = dx  # pixel size in the z direction TODO: 3-D images
    DVol = dx * dy * dz  # element volume    

    originX = np.ceil(nx / 2) # Set image region origin in the x direction
    originY = np.ceil(ny / 2) # Set image region origin in the y direction
    y, x = ind2sub([nx, ny], np.linspace(0, N - 1,N)) # nornalized coordinate pixel position [x]=(N,); [y]=(N,) 
    rj = np.array([(x - originX) * dx, (y - originY) * dy, np.zeros((len(x)))]) # pixel position [rj]=(3,N)
    
    # SENSOR DISCRETIZATION (for integrating line detectors in the z direction)
    posSensLin = posSens  # posSensLin(:,:,i) position of the surface elements (treated as point detectors) of the divided "i" sensor
    if nls > 3:
        posz = np.linspace(-ls / 2, ls / 2, nls) # position of the surface elements (treated as point detectors) of the divided sensor
        posz = np.reshape(posz, (len(posz), 1))  
        posSensLin = np.reshape(posSens, (1, 3, Ns))
        aux = posSensLin
        for k in range(1, len(posz)):
            posSensLin = np.vstack((posSensLin, aux))
        posSensLin[0:, 2, 0:] = np.matlib.repmat(posz, 1, Ns)
    else:
        posz = 0
        posz = np.array([posz])
        posSensLin = np.reshape(posSens, (1, 3, Ns))
        posSensLin[0:, 2, 0:] = np.matlib.repmat(posz, 1, Ns)

    # TIME GRID
    dt = tt[1] - tt[0] # time step at which velocity potencials are sampled
    to = int(tt[0] / dt)
    tf = int(tt[len(tt) - 1] / dt) + 1
    sampleTimes = np.arange(to, tf, dtype=int)
    Nt = len(sampleTimes)
    
    # SPATIAL IMPULSE RESPONSE (SIR) MATRIX
    print('Creating SIR Matrix...'); # describes the spreading of a delta pulse over the sensor surface
    Gs = np.zeros((Ns*Nt, N)) # [1/m]
    currentSens = 0  # Current sensor
    currentTime = 0  # Current time
    for i in tqdm(range(1, np.size(Gs, 0) + 1)):  # Python processing by rows is faster and more efficient
        acum = np.zeros((1, np.size(Gs, 1))) # sum of the velocity potencial detected by each of the surface elements of the divided sensor
        for kk in range(0, nls):  # For each surface element of the divided sensor
            # Calculate the distance between DVol and posSensLin(:,:,i) 
            aux = np.reshape(posSensLin[kk, 0:, currentSens], (3, 1)) @ np.ones((1, N))
            aux2 = rj - aux
            R = np.sqrt(aux2[0, 0:] ** 2 + aux2[1, 0:] ** 2 + aux2[2, 0:] ** 2)
            R = np.reshape(R, (1, len(R)))
            # All non-zero elements in a row A belong to DVol whose centers lie within a spherical shell of radius vs*tk and width vs*dt around posSens:
            # delta = 1   si    |t_k - R/vs| < dt/2
            #         0         en otro caso
            delta = (np.abs(sampleTimes[currentTime] * dt - R / vs) <= dt / 2) 
            delta = delta * 1  # Bool to int
            acum = acum + delta / R  # sum of the velocity potencial detected by each of the surface elements of the divided sensor
        Gs[i - 1, 0:] = acum
        currentTime = currentTime + 1  
        if np.mod(i, Nt) == 0: # Calculate for other sensor
            currentSens = currentSens + 1
            currentTime = 0
    
    # LASER PULSE DURATION EFFECT AND SHOT NOISE REDUCTION
    if rsnoise: # Reduce shot noise induced by the arrival of wave from individual volume elements
        tprop = DVol/(3*vs) # Average sound propagation time through a volume element
    
        if tprop<tlp:
            tprop = tlp # if tp > tprop, accurate modeling of Gs requires the use of a broader Gaussian function convolution
        
        Ti = np.arange(-np.ceil(Nt/2),np.ceil(Nt/2),dtype=int)*dt
        Gi = 2/(tprop*np.sqrt(np.pi))*np.exp(-1*((2*Ti/tprop)**2))
        Gi = convolution_matrix(Gi,Nt,'same')
        Gi = sparse.kron(csc_matrix(np.eye(Ns)),csc_matrix(Gi))
        Gs = Gi@Gs    
    
    # SYSTEM MATRIZ
    if tdo:
        print('Creating PA Matrix...'); # describes the specific temporal signal from a PA point source
        print('Applying Time Derivative Operator...');
        #Tm=toeplitz(np.arange(0,Nt),np.arange(0,-Nt,-1));
        Tm=np.arange(-np.ceil(Nt/2),np.ceil(Nt/2),dtype=int)
        Tm2=(np.abs(Tm)<=(dx/(vs*dt)))*1; 
        Tm2=Tm2*Tm
        Tm2=Tm2*(-vs*dt/(2*dx));
        Gpa=convolution_matrix(Tm2,Nt,'same') # GPa is adimensional
        Gpa=sparse.kron(csc_matrix(np.eye(Ns)),csc_matrix(Gpa))

        A = Gpa@Gs
        A = (Betta/(4*np.pi*vs**2)*DVol/dt**2)*A;  # A is adimensional
    else:
        A = Gs
        A = (-Betta / (4 * np.pi * rho * Calp) * DVol / dt) * A  # [m^5/(J*s)]    
    
    if normA: # System Matrix Normalization
        A = A / np.max(np.abs(A.ravel())) 
    
    if thresh>0: # Threshold the matrix to remove small entries and make it more sparse
        indumbral=np.where(np.abs(A)<10**(-thresh))
        A[indumbral]=0
    
    return A

###############################################################################
def build_ps_matrix(nx, dx, dsa, posSens, vs, tt):
    """
    Model-based Matrix by Pseudo-Spectral Approach -> A: (Ns*Nt,N)
    (Similar to k-Wave)
    
    nx: number of pixels in the x direction for a 2-D image region
    dx: pixel size  in the x direction [m]
    dsa: distance sensor array [m]
    posSens: position of the center of the detectors (3,Ns) [m]
    vs: speed of sound (homogeneous medium) [m/s]
    tt: time samples (Nt,) [s]
    
    References:
        [1] K. Francis, et al., "A simple and accurate matrix for model based
        photoacoustic imaging", IEEE Proc. of the Healthcom (2016).
    """
    
    # TODO: mejorar calculo grilla de capa absorbedora en funcion de la imagen y el tiempo.
    
    # 2-D IMAGE REGION GRID
    ny = nx  # nx = ny. Rectangular grid
    N = nx * ny  # Total pixels 
    dy = dx  # pixel size in the y direction
    
    y, x = ind2sub([nx, ny], np.linspace(0, N - 1,N)) # nornalized coordinate pixel position [x]=(N,); [y]=(N,) 
    originX = np.ceil(nx / 2) # Set image region origin in the x direction
    originY = np.ceil(ny / 2) # Set image region origin in the y direction
    ri = np.array([(x - originX) * dx, (y - originY) * dy]) # pixel position [ri]=(2,N)
    
    # TIME GRID
    dt = tt[1] - tt[0] # time step at which velocity potencials are sampled
    to = int(tt[0] / dt)
    tf = int(tt[len(tt) - 1] / dt) + 1
    sampleTimes = np.arange(to, tf, dtype=int) # (Nt,)
    Nt = len(sampleTimes)
    sampleTimes = np.reshape(sampleTimes,(1,Nt))
    
    # COMPUTATIONAL GRID
    Nout = np.round(dsa*2/dx)  # number of grid points in the x or y direction 
    # Adding grid points to avoid re-entering reflecting waves (matched layer)
    Nout1 = int(Nout + np.ceil(nx/2) + 1)
    Nout2= int(np.ceil(tt[-1]*vs/dx))
    if Nout1>Nout2:
        Nout=Nout1
    else:
        Nout=Nout2
    # Choose next power of 2 greater than Nout
    pow2 = 1; pow2=int(pow2)
    while(pow2 < Nout):
        pow2*=2
    Nout = pow2
    originX = np.ceil(Nout / 2) # Set image region origin in the x direction
    originY = np.ceil(Nout / 2) # Set image region origin in the y direction
    Nout2 = Nout**2
    ky, kx = ind2sub([Nout, Nout], np.linspace(0, Nout2 - 1,Nout2))
    kxy = 2*np.pi/(Nout*dx)*np.array([(kx - originX), (ky - originY)]) # spatial frequency k=2*pi/r [kxy]=(2,Nout2)
    
    # Forward Discrete Fourier Transform
    kxy = np.transpose(kxy) # [kxy]=(Nout2,2)
    
    Wfwd = 1/Nout*np.exp(-1j*kxy@ri) # [Wfwd]=(Nout2,N) Complex Matrix
    
    # Sensor positions
    rs=posSens[0:2,:] # only x and y directions [rs]=(2,Ns)
    Ns = posSens.shape[1] # number of detectors
    
    # Free memory space
    del y,x,ri,ky,kx
    
    # Calculate matrix that combines propagation and Fourier inversion
    km = np.linalg.norm(kxy,ord=2,axis=1) # (Nout2,)
    km = np.reshape(km,(Nout2,1))
    for i1 in tqdm(range(0,Ns)):
        Winv = 1/Nout*np.exp(-1j*kxy@rs[:,i1]) #(Nout2,1)
        Winv = np.transpose(Winv) # [Winv]=(1,Nout2) Complex Matrix
        kapat = np.cos((vs*km)@(sampleTimes*dt)) # (Nout2,Nt)
        kapat = np.transpose(kapat) # (Nt,Nout2)
        Kt = Winv * kapat # (Nt,Nout2)
        if i1==0:
            K=Kt
        else:
            K=np.vstack((K,Kt)) # (Ns*Nt,Nout2)
            
    # Free memory space
    del kxy,kapat,Winv,Kt
    
    # SYSTEM MATRIX
    A = K@Wfwd   # [A] = (Ns*Nt,N)
    A = np.real(A)
    
    return A
    
###############################################################################    
import pickle
from collections import deque

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from PIL import Image
import skimage.morphology, skimage.filters
from scipy.ndimage import convolve1d,label, gaussian_filter
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator


"""
This class generates filaments and realisitcally mimics the imaging process of a flourescence microscope
to generate realistic training data for the skeletonisation model.
"""
class FMGenerator:

    nfil = 1
    resolution = (80,80,8)
    noise = 0.0
    blur = 1.0
    back = 0.1

    def __init__(self, n,res,err,blur,back,backnoise=0.0,lowfreq=0.0,filnoise=0.0):
        # res should be a 3-touple: x,y,angular resolution
        self.nfil = n
        self.resolution = res
        self.noise = err
        self.blur = blur
        self.back = back
        self.backn = backnoise
        self.lf = lowfreq
        self.fln = filnoise
    
    def make(self):
        f = np.zeros(self.resolution)

        steps = 400
        phistd = 0.04
        ds = 0.8*self.resolution[0]/steps

        # use a possionized number of filaments
        N = np.random.poisson(self.nfil)
        if N == 0:
            return f
        
        for i in range(N):
            dphi = np.random.normal(loc=0.0,scale=phistd,size=steps)
            phi0 = np.random.uniform(0.0,2*np.pi)
            phi = np.cumsum(dphi)+phi0
            x0 = np.random.uniform(self.resolution[0]/16 , 15*self.resolution[0]/16)
            y0 = np.random.uniform(self.resolution[1]/16 , 15*self.resolution[1]/16)
            dx = ds*np.cos(phi)
            dy = ds*np.sin(phi)
            x = x0 + np.cumsum(dx)
            y = y0 + np.cumsum(dy)
            # add the result into the image
            ix = x.astype(int)
            iy = y.astype(int)
            iphi = (self.resolution[2]*(phi/np.pi)).astype(int) % self.resolution[2]
            for s in range(steps):
                if ((ix[s] >= 0) and (iy[s] >= 0)) and ((ix[s] < self.resolution[0]) and (iy[s] < self.resolution[1])):
                    f[ix[s],iy[s],iphi[s]] += 1.0
        # normalize the image
        f *= self.resolution[0]/steps
        # apply soft convolution
        f = gaussian_filter(f,sigma=(0.5,0.5,0.4),mode='wrap')

        return f

    def forward_operator(self,f):
        fs = np.sum(f,axis=2)
        g = fs+self.back
        # poisson noise
        m = 12
        g = np.random.poisson(m*g)/m
        # apply blur
        g = gaussian_filter(g,self.blur)
        # apply noise level
        e = np.random.normal(scale=self.noise,size=g.shape)
        # intentisity proportional filament noise
        ef = np.random.normal(size=g.shape)*np.sqrt(fs)*self.fln

        # optional apply low frequency background effects
        dback = self.backn*0.5*(np.random.uniform()+1)
        
        nback = np.clip(self.lf*20.0*gaussian_filter(np.random.normal(size=(self.resolution[0],self.resolution[1])),sigma=20.0,mode='wrap'),0,np.inf)

        gobs = np.clip(g + e + ef + dback + nback,0,np.inf)
        return gobs


#### PARAMETERS FOR CNN MODEL
# this defines the model we use
standard_resolution = (80,80,16)
# do we use a bias in the covolutions? recommended: False
BIAS = False

"""
This is a fully convolutional network, that which can be used, after training as follows:
- saving the model: model.save(path)
- loading the model: tf.keras.models.load_model(path)
- applying the model to a flourescence image directly: model.encoder(data as 4-d array [n_img,xres,yres,1])
# note: use_bias either all True or all False, both has advantages/disadvantages
"""
class FilamentReconstructor(Model):
    # stores latest model for default use
    latest = None

    def __init__(self, resolution):
        super(FilamentReconstructor, self).__init__()   
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(resolution[0],resolution[1],1)),
            layers.Conv2D(64, (5,5), activation='relu', padding='same', strides=1, use_bias=BIAS),
            layers.Conv2D(32, (9, 9), activation='relu', padding='same', strides=1, use_bias=BIAS),
            layers.Conv2D(64, (15,15), activation='relu', padding='same', strides=2, use_bias=BIAS),
            layers.Conv2DTranspose(32, kernel_size=15, strides=2, activation='relu', padding='same',use_bias=BIAS),
            layers.Conv2DTranspose(32, kernel_size=9, strides=1, activation='relu', padding='same',use_bias=BIAS),
            layers.Conv2D(resolution[2], kernel_size=5, activation='relu',strides=1, padding='same',use_bias=BIAS),
        ])
        FilamentReconstructor.latest = self

    def call(self, x):
        restored = self.encoder(x)
        return restored
    
    # static method generates N images and filament configurations according to the imaging model specified in the FMGenerator params
    def gen_training_data(N,resolution,image_model_params=None):

        # image_model_params should be a dictionary containing the necessary parameters for the FMGenerator construction
        params = {'n': 15, 'res':resolution, 'err': 0.015, 'blur':1.0, 'back':0.02,'backnoise':0.05,'lowfreq':0.05,'filnoise':0.07}
        if isinstance(image_model_params,dict):
            params = image_model_params
        G = FMGenerator(**params)

        Y = np.zeros((N,resolution[0],resolution[1],resolution[2]))
        X = np.zeros((N,resolution[0],resolution[1]))
        for i in range(N):
            f = G.make()
            Y[i,:,:,:] = f 
            X[i,:,:] = G.forward_operator(f)
        return X,Y

    """
    Returns a model which solves the MT inverse problem for a specified imaging forward model,
    in this case defined via the prarameters 
    """
    def train_model(Ndata,resolution,epochs,image_model_params=None):
        X,Y = FilamentReconstructor.gen_training_data(Ndata,resolution,image_model_params)

        ntrain = int(0.9*Ndata)
        
        Xtrain = X[:ntrain,:,:]
        Xtrain = Xtrain[...,tf.newaxis]
        Ytrain = Y[:ntrain,:,:,:]

        Xtest = X[ntrain:,:,:]
        Xtest = Xtest[...,tf.newaxis]
        Ytest = Y[ntrain:,:,:,:]

        recon = FilamentReconstructor(resolution)
        recon.compile(optimizer=Adam(1e-3), loss=losses.MeanSquaredError())

        recon.fit(Xtrain, Ytrain,
                    epochs=epochs,
                    shuffle=True,
                    validation_data=(Xtest, Ytest))
        
        ## create a table of reference quantiles for rescaling of input images
        if BIAS:
            N = np.linspace(0.0,60.0,num=61)
            QT = np.zeros((610,3))

            for i in range(61):
                image_model_params['n'] = N[i]
                G = FMGenerator(**image_model_params)
                for k in range(10):
                    f = np.asfarray(G.forward_operator(G.make()))
                    QT[10*i+k,:] = np.quantile(f.flatten(),[0.01,0.6,0.99])
            recon.QT = QT

        return recon

    # not relevant for a homeogenous (no bias) network
    def rescale(I,QT):
        R = (QT[:,2]-QT[:,1])/(QT[:,1]-QT[:,0])
        ql,qm,qh = np.quantile(I.flatten(),[0.01,0.6,0.99])
        r = (qh-qm)/(qm-ql)
        rsi = np.argmin(np.abs(R-r))
        qtl = QT[rsi,0]
        qth = QT[rsi,2]
        In = (I-ql)/(qh-ql)
        In = qtl+(qth-qtl)*In
        return In

    def apply_model(self,image,size=80,angles=16):
        # slice the image
        stride = 20
        eff_size = size - 2*stride
        height,width = image.shape
        nx = (height-2*stride) // eff_size
        ny = (width-2*stride) // eff_size
        xoff = ((height-2*stride) % eff_size) // 2 + stride
        yoff = ((width-2*stride) % eff_size) // 2 + stride
        x0 = np.linspace(0,eff_size*(nx-1),num=nx,dtype=int)
        y0 = np.linspace(0,eff_size*(ny-1),num=ny,dtype=int)
        ims = np.zeros((nx*ny,size,size))
        
        ind = 0
        for ix in range(nx):
            for iy in range(ny):
                piece = image[xoff+x0[ix]-stride:(xoff+x0[ix]+eff_size)+stride,yoff+y0[iy]-stride:(yoff+y0[iy]+eff_size)+stride]
                if BIAS:
                    ims[ind,:,:] = FilamentReconstructor.rescale(piece,self.QT)
                else:
                    ims[ind,:,:] = piece
                ind += 1
        
        solim = self.encoder(ims[:,:,:,None]).numpy()
        
        recover = np.zeros((nx*eff_size,ny*eff_size,angles))
        
        ind = 0
        for ix in range(nx):
            for iy in range(ny):
                recover[ix*eff_size:(ix+1)*eff_size,iy*eff_size:(iy+1)*eff_size,:] = solim[ind,stride:-stride,stride:-stride,:]
                ind += 1
        
        return recover

    def load_model(location):
        mod = tf.keras.models.load_model(location)
        FilamentReconstructor.latest = mod
        return mod

class SegmentUtility:
    def skeletonize_2p1d(Ip):
        # first we apply a local threshold maks
        T = skimage.filters.threshold_local(Ip,block_size=(5,5,17),mode='wrap')
        T2 = skimage.filters.threshold_li(Ip,initial_guess=20.0)
        # tunable paramters..
        mask = (Ip>(2.7*T))*(Ip>T2)
        
        # pad the mask due to perdiodicy in p
        p = mask.shape[2]
        pmask = np.zeros((mask.shape[0],mask.shape[1],p*3))
        pmask[:,:,:p] = mask
        pmask[:,:,p:2*p] = mask
        pmask[:,:,2*p:3*p] = mask

        S = skimage.morphology.skeletonize_3d(pmask)
        S=skimage.morphology.remove_small_objects(np.array(S,bool),12,connectivity=3)
        return S[:,:,p:2*p]

    def find_neighbors(S):
        res = S.shape
        ker = np.ones(3)
        # convolve in 3 dimensions
        Q = convolve1d(np.array(S,dtype=np.int32),ker,axis=2,mode='wrap')
        Q = convolve1d(Q,ker,axis=1,mode='wrap')
        Q = convolve1d(Q,ker,axis=0,mode='wrap')
        # remove the non skelett fields
        Q -= 1
        Q = np.array(Q*S,dtype=np.int32)
        return Q

    def label_filaments(S):
        # compute the neighbors
        N = SegmentUtility.find_neighbors(S)
        a = S.shape[2]
        res = S.shape[2]-1
        # label filament excluding intersection endpoints
        L,_ = label((0<N)*(N<=2),structure=np.ones((3,3,3)))
        # account for periodicity, match labels
        xb,yb = np.nonzero(N[:,:,0]==2)
        for i in range(len(xb)):
            l = 0
            x = xb[i]
            y = yb[i]
            lb = L[x,y,0]
            for j in [-1,0,1]:
                for k in [-1,0,1]:
                    if N[x+j,y+k,res] == 2:
                        l = L[x+j,y+k,res]
            if (l != 0) and (lb != l):
                L[L==l] = lb
        # account for filament endpoints at crossings:
        N1 = SegmentUtility.find_neighbors(L>0)
        # filament endpoints
        EP = (N1==1)
        # coordinates of filament endpoints
        xe,ye,ze = np.nonzero(EP)
        ne = len(xe)
        for e in range(ne):
            x = xe[e]
            y = ye[e]
            z = ze[e]
            l = L[x,y,z]
            # for each endpoint find possible close by crossing and label it accordingly
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    for k in [-1,0,1]:
                        if N[(x+i),(y+j),(z+k)%a]>2:
                            L[(x+i),(y+j),(z+k)%a]=l
        # TODO: Piece angled filaments back together if there are no possible conflicts..
        # All filament intersection and ends
        SEP = (N1 > 0)*(N1 != 2)
        return L,SEP

# stores the coordinates of a single filament and all other parameters of interest
class Curve:
    
    def fit_circle_cost(p,xt,yt,x,y,w):
        # p parameter: p[0]: curvature, p[1]: tangential angle
        # decide if curvature == 0
        if(p[0] != 0.0):
            # cost circle
            x0 = xt*p[0]+np.cos(p[1])
            y0 = yt*p[0]+np.sin(p[1])
            c = (np.sqrt((p[0]*x-x0)**2+(p[0]*y-y0)**2)-1.0)**2*w/(p[0]**2)
        else:
            #cost line
            vx = -1*np.sin(p[1])
            vy = np.cos(p[1])
            Sx = x-xt
            Sy = y-yt
            t = vx*Sx+vy*Sy
            c = (Sx-vx*t)**2+(Sy-vy*t)**2
        return np.sum(c)
    
    def fit_circle_cost_jac(p,xt,yt,x,y,w):
        # p parameter: p[0]: curvature, p[1]: tangential angle
        # decide if curvature == 0
        g = np.ones(2)
        if(p[0] != 0.0):
            # cost circle
            x0 = xt*p[0]+np.cos(p[1])
            y0 = yt*p[0]+np.sin(p[1])
            qq = np.sqrt((p[0]*x-x0)**2+(p[0]*y-y0)**2)
            c = (qq-1.0)**2*w/(p[0]**2)
            aa = w/p[0]**2
            xx = p[0]*x-x0
            yy = p[0]*y-y0
            g[0] = np.sum(-2.0*c/p[0]+2*aa*(1.0-1.0/qq)*((p[0]*x-x0)*(x-xt)+(p[0]*y-y0)*(y-yt)))
            g[1] = np.sum(2*aa*(1.0-1.0/qq)*((p[0]*x-x0)*np.sin(p[1])-(p[0]*y-y0)*np.cos(p[1])))
        else:
            #cost line
            vx = -1*np.sin(p[1])
            vy = np.cos(p[1])
            Sx = x-xt
            Sy = y-yt
            t = vx*Sx+vy*Sy
            c = (Sx-vx*t)**2+(Sy-vy*t)**2
        return (np.sum(c),g)
    
    def fit_circle(xt,yt,x,y,s,p0g=None,p1g=None):
        # sqrt(s) is the scale!
        w = np.exp(-((x-xt)**2+(y-yt)**2)/(2*s))
        w /= np.sum(w)
        if(p0g == None):
            res1 = minimize(Curve.fit_circle_cost_jac,x0=[0.01,0.1],args=(xt,yt,x,y,w),tol=10e-5,jac=True)
            res2 = minimize(Curve.fit_circle_cost_jac,x0=[-0.01,0.1],args=(xt,yt,x,y,w),tol=10e-5,jac=True)
        else:
            res1 = minimize(Curve.fit_circle_cost_jac,x0=[p0g,p1g],args=(xt,yt,x,y,w),tol=10e-5,jac=True)
            res2 = minimize(Curve.fit_circle_cost_jac,x0=[-p0g,p1g],args=(xt,yt,x,y,w),tol=10e-5,jac=True)
        if(res1.fun > res2.fun):
            return res2.x
        else:
            return res1.x
        
    def cp_analysis(self,s):
        #sqrt(s): scale
        """ Calculates orientation and curvature for each point of the curve using a circle fitting methods and a Gaussian Kernen for weighting of neighbour point importance.
        s is the variance of said kernel. 
        Parameters
        ----------
        s: variance of the Gaussian Kernel

        Returns
        -------
        Nothing but self.k stores the curvature estimates and self.spol the orientation as angle in [0,2*pi]
        """
        d = 2*s
        self.cpa = True
        self.k = np.zeros(self.N)
        self.spol = np.zeros(self.N)
        for i in range(self.N):
            lb = max(0,i-d)
            ub = min(d+i+1,self.N-1)
            xf = self.x[lb:ub]
            yf = self.y[lb:ub]
            if (i == 0):
                p = Curve.fit_circle(self.x[i],self.y[i],xf,yf,s)
            else:
                p = Curve.fit_circle(self.x[i],self.y[i],xf,yf,s,self.k[i-1],self.spol[i-1])
            self.k[i] = p[0]
            self.spol[i] = p[1]
        self.avgk = np.average(np.absolute(self.k))
        self.maxK = np.max(np.absolute(self.k))
        return

    def __init__(self,x,y,o):
        """
        Generate a Curve object by giving the x and y coordinates as 1d-arrays, also o is the orientation estimation in (0,pi) for each pixel
        """
        self.N = len(x)
        self.x = x
        self.y = y
        self.pol = o
        if self.N >= 2:
            if(self.N >= 3):
                self.vx = np.gradient(x,edge_order=2)
                self.vy = np.gradient(y,edge_order=2)
            if(self.N == 2):
                self.vx = np.gradient(x)
                self.vy = np.gradient(y)
            self.v = np.sqrt(self.vx**2+self.vy**2)
            self.length = np.sum(self.v)
        
    def average(self,H):
        return np.average(H[self.x,self.y])
    

    def refine(self, Img):
        sigma = 5.0
        x0 = self.x
        y0 = self.y
        offset = 5
        frame = [np.min(x0)-offset,np.max(x0)+offset,np.min(y0)-offset,np.max(y0)+offset]
        V = gaussian_filter(Img[frame[0]:frame[1],frame[2]:frame[3]],sigma=1.2)
        V /= np.max(V)
        xF = x0-frame[0]
        yF = y0-frame[2]
        xStart = xF[0]
        xEnd = xF[-1]
        yStart = yF[0]
        yEnd = yF[-1]
        # build V interpolation
        Xg = np.array(range(0,frame[1]-frame[0]))
        Yg = np.array(range(0,frame[3]-frame[2]))
        field = RegularGridInterpolator((Xg,Yg),V,'cubic',False,0.0)


        def naive_interpolation(steps):
            du = 1.0/(steps+1)
            u = np.linspace(du,1.0-du,num=steps)
            # transform to length
            dL = np.sqrt((xF[1:]-xF[:-1])**2 + (yF[1:]-yF[:-1])**2)
            L = np.sum(dL)
            l = L*u
            iL = np.cumsum(dL)
            niX = np.zeros((steps))
            niY = np.zeros((steps))
            seg = 1
            for i in range(steps):
                while iL[seg-1] <= l[i]:
                    seg += 1
                # interpolate
                alpha = (iL[seg-1]-l[i])/dL[seg-1]
                niX[i] = alpha*xF[seg-1]+(1.0-alpha)*xF[seg]
                niY[i] = alpha*yF[seg-1]+(1.0-alpha)*yF[seg]
            return niX,niY,du*L

        # start with a naive coordinate interpolation
        factor = 5
        interX,interY,dL = naive_interpolation(factor*(self.N-1))
        n = len(interX)

        k = 1.5
        dx = 1e-3
        Cbuf = np.zeros((n,2))
        delx = np.zeros((n,2))
        delx[:,0] = dx
        dely = np.zeros((n,2))
        dely[:,1] = dx

        def energy(P):
            k = 10.0
            dl = P[0]
            X = P[1:n+1]
            Y = P[n+1:]
            # calculate streching energy
            E = np.sum( (np.sqrt((X[1:]-X[:-1])**2 + (Y[1:]-Y[:-1])**2) - dl)**2)
            # add endpoints
            E += (np.sqrt((X[0]-xStart)**2+(Y[0]-yStart)**2) - dl)**2 + (np.sqrt((X[-1]-xEnd)**2+(Y[-1]-yEnd)**2) - dl)**2
            # constant
            E *= k
            # field contribution
            ep = np.array([X,Y])
            E -= np.sum(field(ep.T))
            E += 2.0*(dl-dL)**2
            return E
        
        def energy_2(P):
            X = P[:n]
            Y = P[n:]
            # calculate streching energy
            E = np.sum((X[1:]-X[:-1])**2 + (Y[1:]-Y[:-1])**2)
            # add endpoints
            E += ((X[0]-xStart)**2+(Y[0]-yStart)**2) + ((X[-1]-xEnd)**2+(Y[-1]-yEnd)**2)
            # constant
            E *= k
            # field contribution
            ep = np.array([X,Y])
            E -= np.sum(field(ep.T))
            return E
        
        def energy_2_jac(P):
            X = P[:n]
            Y = P[n:]
            dX = X[1:]-X[:-1]
            dY = Y[1:]-Y[:-1]
            G = np.zeros(2*n)
            G[0:n-1] += dX
            G[1:n] -= dX
            G[n:-1] += dY
            G[n+1:] -= dY
            # end points
            G[0] -= X[0]-xStart
            G[n-1] += xEnd-X[n-1]
            G[n] -= Y[0]-yStart
            G[-1] += yEnd-Y[n-1]
            G *= 2*k
            # field parameters ..
            Cbuf[:,0] = X
            Cbuf[:,1] = Y
            Fb = field(Cbuf)
            Fx = field(Cbuf+delx)
            Fy = field(Cbuf+dely)
            G[:n] += -(Fx-Fb)/dx
            G[n:] += -(Fy-Fb)/dx
            return -G

        
        res = minimize(energy_2,np.concatenate([interX,interY]),method="CG", jac=energy_2_jac)
        refX = np.zeros((n+2))
        refY = np.zeros((n+2))
        refX[0] = xStart
        refX[-1] = xEnd
        refY[0] = yStart
        refY[-1] = yEnd
        refX[1:-1] = res.x[:n]
        refY[1:-1] = res.x[n:]
        refX += frame[0]
        refY += frame[2]
        self.xold = self.x
        self.yold = self.y
        self.nold = self.N
        self.x = refX
        self.y = refY
        self.N = n+2
        return refX,refY

class Filament:

    def util_swap(X,Y,O,a,b):
        xt = X[a]
        yt = Y[a]
        ot = O[a]
        X[a] = X[b]
        Y[a] = Y[b]
        O[a] = O[b]
        X[b] = xt
        Y[b] = yt
        O[b] = ot
    
    def util_pdist(r1,r2,rot):
        a = max(r1,r2)
        b = min(r1,r2)
        return min(a-b,b+rot-a)

    def trace(self,X,Y,O,rot=16):
        d = deque([0])
        blocked = np.zeros(len(X))
        pivp = 0
        pivm = 0
        for t in range(1,len(X)):
            for i in range(1,len(X)):
                if blocked[i]:
                    continue
                if np.abs(X[i]-X[pivp]) <= 1:
                        if np.abs(Y[i]-Y[pivp]) <= 1:
                            if Filament.util_pdist(O[i],O[pivp],rot) <= 1:
                                pivp = i
                                d.append(pivp)
                                blocked[i] = 1
                                break
                if np.abs(X[i]-X[pivm])<= 1:
                        if np.abs(Y[i]-Y[pivm]) <= 1:
                            if Filament.util_pdist(O[i],O[pivm],rot) <= 1:
                                pivm = i
                                d.appendleft(pivm)
                                blocked[i] = 1
                                break
        idx = list(d)
        return Curve(X[idx],Y[idx],O[idx])

    """
    F: A labeled 2+1d array witht the different filaments
    """
    def __init__(self,Fx,Ep):
        r = Fx.shape[2]
        self.cvcount = np.max(Fx)
        self.curves = []
        for t in range(1,self.cvcount+1):
            X,Y,O = np.nonzero(Fx==t)
            if len(X)==0:
                continue
            #cv = self.follow(X,Y,O,spi)
            cv = self.trace(X,Y,O,rot=r)
            self.curves.append(cv)
        self.nodes = Ep
        self.nodecount = np.sum(Ep)

    def cp_analysis(self,s):
        # s: scale
        for cv in self.curves:
            cv.cp_analysis(s)
        return

class CNNFilamentator:
    """
    Creates a CNNSegmentor objects which can be used to detect and label the filaments.
    model: directory of the stored trained model to use.
    """
    def __init__(self, model=None):
        if model is None:
            self.cnn = FilamentReconstructor.latest
            if self.cnn == None:
                print("[ERROR] No default model found. Please specify a model path or a specific FilamentReconstructor model.")
            return
        if isinstance(model,FilamentReconstructor):
            self.cnn = model
            return
        self.cnn = tf.keras.models.load_model(model)

    """
    Takes an image and returns a labeled 2+1D image containing the identified filaments
    """
    def extract_filaments(self,MT):
        mtx = FilamentReconstructor.apply_model(self.cnn,MT)
        Sx = SegmentUtility.skeletonize_2p1d(mtx)
        Fx,Ep = SegmentUtility.label_filaments(Sx)
        F = Filament(Fx,Ep)
        return Sx,Fx,F

# this class stores a Filament object and generates this automatically from fluorescence microskopy images 
# can be saved and reloded to hard drive to prevent rerunning calculations
# stores all relevant informations about a sample cell
class Cell:
    def __init__(self,MT,metadata,cnn_model=None, verbose=False):
        '''
            MT: Microtuble channel
           # IF: IF channel # currently out
           # C3: some aditional channel (pattern, core)
            metadata: dictionary containing at least items: 'date','pattern','type','resolution','c3'
            cnn_proc: The FilamentReconstructor model for filament extraction, or a path to load it, or None: use default
        '''
        if verbose:
            print(f"[INFO] on cell {metadata}: applying CNN.")
        cnnFil = CNNFilamentator(cnn_model)
        self.metadata = metadata
        self.MT = MT-np.min(MT) # just add the MT channel
        # extract and process filaments
        if verbose:
            print(f"[INFO] on cell {metadata}: segmenting CNN output.")
        self.skelett,self.labeled,self.filament = cnnFil.extract_filaments(MT)
        # do the curvature analysis use 16 px**2 as scale
        if verbose:
            print(f"[INFO] on cell {metadata}: refining filaments.")
        rfc = 0
        for cv in self.filament.curves:
            rfc += 1
            if cv.N >= 5:
                cv.refine(MT)
            if rfc % 100 == 0:
                print(rfc)
        if verbose:
            print(f"[INFO] on cell {metadata}: calculating refined curvature/orientation.")
        self.filament.cp_analysis(25)
        return

    def save(self,path):
        with open(path, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        return

    def load(path):
        with open(path, 'rb') as handle:
            c = pickle.load(handle)
        return c
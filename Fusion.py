from pfapack import pfaffian as pf
import numpy as np
import numpy.linalg as nla
from scipy.integrate import  odeint,solve_ivp
from numpy.fft import fft, fftshift, ifft, ifftshift
from scipy import interpolate
class Fusion:
    _eps=1e-10
    def __init__(self,E,t,Delta,T=100,sigma=None,sigma_t=None,noise_type='',seed=None,scaling=2,t_f=[0]*3,Delta_f=[0]*3,E_f=[0]*4):
        '''
        `sigma` controls the fluctuation of disorder
        `sigma_t` controls the temporal correlation in terms of omega, where the cutoff of the angular frequency  is at 2*pi/sigma_t
        `noise_type` = E | D (Delta) | t
        '''
        self.E=[E*1.]
        self.t=[t*1.]
        self.Delta=[Delta*1.]
        self.t_f=np.array(t_f)
        self.Delta_f=np.array(Delta_f)
        self.E_f=np.array(E_f)
        self.T=T
        if sigma is not None:
            self.sigma=sigma
        if sigma_t is not None:
            self.sigma_t=sigma_t
        self.noise_type=noise_type
        self.seed=seed
        self.scaling=scaling
        self.N_dot=E.shape[0]
        self.t_list=[0]
        self.prev_dE=0
        if not self.noise_type=='':
            self.get_noise()
        self.M_ee = np.array([
            [0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, -1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0]
        ])
        self.M_eo = np.array([
            [0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0]
        ])
        self.f_list3=np.array([1,1,1],dtype=np.float64)
        self.f_list4=np.array([1,1,1,1],dtype=np.float64)
        # For majorana basis
        self.O=np.zeros((self.N_dot,self.N_dot),dtype=np.float64)
        self.ix=np.arange(self.N_dot)

    def get_BdG(self,t_ind=-1):
        H_0=np.zeros((self.N_dot,self.N_dot),dtype=np.float64)
        H_0[self.ix,self.ix]=self.E[t_ind]
        H_0[self.ix[:-1],self.ix[:-1]+1]=self.t[t_ind]
        H_0[self.ix[:-1]+1,self.ix[:-1]]=self.t[t_ind]

        Delta_matrix=np.zeros((self.N_dot,self.N_dot),dtype=np.float64)
        Delta_matrix[self.ix[:-1],self.ix[:-1]+1]=self.Delta[t_ind]
        Delta_matrix[self.ix[:-1]+1,self.ix[:-1]]=-self.Delta[t_ind]

        h_BdG=np.empty((2*self.N_dot,2*self.N_dot),dtype=np.float64)
        h_BdG[:4,:4]=H_0
        h_BdG[4:,4:]=-H_0.T
        h_BdG[:4,4:]=Delta_matrix
        h_BdG[4:,:4]=Delta_matrix.T
        return h_BdG
    
    def get_A(self,t_ind=-1):
        '''Basis ordered as \gamma_{1->4}\bar{\gamma}_{1->4}'''
        self.O[self.ix,self.ix]=self.E[t_ind]
        self.O[self.ix[:-1],self.ix[:-1]+1]=self.t[t_ind]-self.Delta[t_ind]
        self.O[self.ix[:-1]+1,self.ix[:-1]]=self.t[t_ind]+self.Delta[t_ind]
        h_Maj=np.zeros((2*self.N_dot,2*self.N_dot),dtype=np.float64)
        h_Maj[:4,4:]=self.O
        h_Maj[4:,:4]=-self.O.T
        return h_Maj
    
    def bandstructure(self,basis):
        if 'f' in basis:
            if not hasattr(self, 'ham_f'):
                self.ham_f=self.get_BdG()
            val,vec=nla.eigh(self.ham_f)
            sortindex=np.argsort(val)
            self.val_f=val[sortindex]
            self.vec_f=vec[:,sortindex]
        if 'm' in basis:
            if not hasattr(self, 'ham_m'):
                self.ham_m=self.get_A()
            val,vec=nla.eigh(self.ham_m)
            sortindex=np.argsort(val)
            self.val_m=val[sortidnex]
            self.vec_m=vec[:,sortindex]

    def correlation_matrix(self):
        if not (hasattr(self, 'val_f') and hasattr(self, 'vec_f')):
            self.bandstructure(basis='f')
        occupancy=np.heaviside(-self.val_f,.5)
        occupancy_mat=np.tile(occupancy,(2*self.N_dot,1))
        self.C_f=(occupancy*self.vec_f)@self.vec_f.T.conj()
    
    def covariance_matrix(self):
        if not hasattr(self, 'C_f'):
            self.correlation_matrix()
        G=self.C_f[:self.N_dot,:self.N_dot]
        F=self.C_f[:self.N_dot,self.N_dot:]
        A=np.zeros((2*self.N_dot,2*self.N_dot),dtype=np.complex128)
        A[:self.N_dot,:self.N_dot]=1j*(F.T.conj()+F+G-G.T)
        A[self.N_dot:,self.N_dot:]=-1j*(F.T.conj()+F-G+G.T)
        A[self.N_dot:,:self.N_dot]=-(np.eye(self.N_dot)+F.T.conj()-F-G-G.T)
        A[:self.N_dot:,self.N_dot:]=-A[self.N_dot:,:self.N_dot].T
        assert A.imag.__abs__().max()<self._eps, 'Covariance matrix non real {:e}'.format(A.imag.__abs__().max())
        self.C_m=(A.real-A.T.real)/2 

    def fusion_protocol(self,t,M_vec,protocol):
        self.t_list.append(t)
        assert protocol in 'AB', 'Unrecognized protocol ({}).'.format(protocol)
        # assert t<=2*self.T, 'Invalid query of time ({:.2f}) which is larger than total duration 2T ({:.2f}).'.format(t,2*self.T)
        scheme={'A':lambda x: self.change_params(x,t_Delta=0,t_E=1),'B':lambda x: self.change_params(x,t_Delta=1,t_E=0)}
        
        M_mat=M_vec.reshape((2*self.N_dot,2*self.N_dot))
        scheme[protocol](t)
        A_t=self.get_A()
        dydt=(A_t@M_mat-M_mat@A_t)
        return ((dydt-dydt.T)/2).flatten()
    
    def change_params(self,t,t_E=0,t_Delta=0):
        '''t_x is the onset time of the decline, t_x=0, start first, t_x=1, start second'''
        self.f_list3[1]=self.f(t/self.T-t_Delta)
        self.f_list4[:]=self.f(t/self.T-t_E)
        # dE=np.array([func(t) for func in self.noises_E]).flatten() if self.noise_type=='E' else np.array([0]*self.N_dot)
        d_E= np.array([np.interp(t,self.ts,noises_E) for noises_E in self.noises_E]) if 'E' in self.noise_type else np.array([0]*self.N_dot)
        d_Delta= np.array([np.interp(t,self.ts,noises_Delta) for noises_Delta in self.noises_Delta]) if 'D' in self.noise_type else np.array([0]*(self.N_dot-1))
        d_t= np.array([np.interp(t,self.ts,noises_t) for noises_t in self.noises_t]) if 't' in self.noise_type else np.array([0]*(self.N_dot-1))
        self.Delta.append((self.Delta[0]-self.Delta_f)*self.f_list3+self.Delta_f+d_Delta)
        self.t.append((self.t[0]-self.t_f)*self.f_list3+self.t_f+d_t)
        self.E.append((self.E[0]-self.E_f)*self.f_list4+self.E_f+d_E)

    def f(self,x):
        # the finite derivative since order of scaling
        if self.scaling==1:
            if x<0:
                return 1
            elif x<1:
                return 1-x
            else:
                return 0
        if self.scaling==2:
            if x<0:
                return 1
            elif x<1:
                return 1-np.sin(.5*np.pi*x)**2 
            else:
                return 0

        if self.scaling==3:
            if x<0:
                return 1
            elif x<1:
                return 1-10*x**3+15*x**4-6*x**5
            else:
                return 0
            
        

    def get_error(self,A,B):
        return 1-np.abs(pf.pfaffian(A+B))/2**4
    
    def get_parity(self,A):
        return A[5,0].real

    def get_noise(self):
        self.ts,_=autocorrelated_noise(T=2*self.T,sigma=self.sigma,sigma_t=self.sigma_t,k=1,seed=self.seed)
        _,self.noises_E=autocorrelated_noise(T=2*self.T,sigma=self.sigma,sigma_t=self.sigma_t,k=self.N_dot,seed=self.seed) if 'E' in self.noise_type else (None,None)

        # tmp=self.noises_E[0]
        # self.noises_E=self.noises_E*0
        # self.noises_E[2]=tmp

        _,self.noises_Delta=autocorrelated_noise(T=2*self.T,sigma=self.sigma,sigma_t=self.sigma_t,k=self.N_dot-1,seed=self.seed) if 'D' in self.noise_type else (None,None)
        
        _,self.noises_t=autocorrelated_noise(T=2*self.T,sigma=self.sigma,sigma_t=self.sigma_t,k=self.N_dot-1,seed=self.seed) if 't' in self.noise_type else (None,None)

        
        # tmp=self.noises_Delta[0]
        # self.noises_Delta=self.noises_Delta*0
        # self.noises_Delta[0]=tmp

    def solve(self,protocol,**kwargs):
        if not hasattr(self, 'C_m'):
            self.covariance_matrix()
        
        # max_step= 0.5*self.sigma_t if self.noise_type is not None else np.inf
        sol=solve_ivp(self.fusion_protocol,(0,self.T*2),self.C_m.flatten(),args=(protocol,), **kwargs)
        M_final=sol.y[:,-1].reshape((8,8))
        error=self.get_error(M_final,self.M_ee if protocol=='A' else self.M_eo)
        parity=self.get_parity(M_final)
        return parity,error
    

def autocorrelated_noise(T,sigma,sigma_t,N=None,seed=None,k=1):
    if N is None:
        N=2*int(2*T/sigma_t)
    assert N%2==0, 'N should be even number to ensure a real time series.'
    dt=2*T/N
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    Y_t=rng.standard_normal((k,N))
    Y_omega=fft(Y_t)
    omega=np.tile(np.arange(-N//2,N//2)/T*2*np.pi,(k,1))
    S_omega=sigma**2*np.sqrt(4*np.pi)*sigma_t*np.exp(-omega**2*sigma_t**2)
    X_omega=np.sqrt(S_omega/dt)*fftshift(Y_omega)
    X_t=ifft(ifftshift(X_omega))
    assert X_t.imag.__abs__().max()<1e-10, 'X_t not real.'
    return np.linspace(0,N,N)/N*T,X_t.real

def log_fit(x,y,x_range=None):
    if x_range:
        if x_range[0]:
            mask=x_range[0]<x
            x=x[mask]
            y=y[mask]
        if x_range[1]:
            mask=x<x_range[1]
            x=x[mask]
            y=y[mask]
    x_log=np.log(x)
    y_log=np.log(y)
    p=np.polyfit(x_log,y_log,1)
    return p

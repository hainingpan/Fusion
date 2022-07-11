from Fusion import *
import argparse
from mpi4py.futures import MPIPoolExecutor
import pickle
import time


def wrapper(inputs):
    protocol,T,sigma,sigma_t,noise_type,scaling,atol,rtol,savedisorder,ensemble_index=inputs
    # print(protocol,T)
    fusion=Fusion(E=np.array([1]*4),t=np.array([1]*3), Delta=np.array([1]*3),T=T,noise_type=noise_type,sigma=sigma,sigma_t=sigma_t,seed=ensemble_index,scaling=scaling)
    parity,error=fusion.solve(protocol=protocol,method='LSODA',atol=atol,rtol=rtol)
    disorder={}
    if savedisorder:
        if hasattr(fusion, 'ts'):
            disorder['ts']=fusion.ts
        if hasattr(fusion, 'noises_E'):
            disorder['noises_E']=fusion.noises_E
        if hasattr(fusion, 'noises_Delta'):
            disorder['noises_Delta']=fusion.noises_Delta
        if hasattr(fusion, 'noises_t'):
            disorder['noises_t']=fusion.noises_t
    return parity,error,disorder

if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('-sigma',type=float,help='sigma for the strength of fluctuation')
    parser.add_argument('-sigma_t',type=float,help='sigma_t for the time correlation')
    parser.add_argument('-noise_type',default='',type=str,help='noise type: E/t/Delta')
    parser.add_argument('-Tmin',default=1.,type=float,help='T min')
    parser.add_argument('-Tmax',default=200.,type=float,help='T max')
    parser.add_argument('-Tnum',default=50,type=int,help='T num')
    parser.add_argument('-ensemble_size',default=2,type=int,help='ensemble size')
    parser.add_argument('-protocol',default='AB',type=str,help='protocol')
    parser.add_argument('-scaling',default=2,type=int,help='scaling')
    parser.add_argument('-atol',default=1e-5,type=float,help='atol, use 1e-5 for disorder/ 1e-10 for no disorder scaling')
    parser.add_argument('-rtol',default=1e-5,type=float,help='rtol, use 1e-5 for disorder/ 1e-10 for no disorder scaling')
    parser.add_argument('-geom',action='store_true',help='use geomspace in T list')
    parser.add_argument('-savedisorder',action='store_true',help='save disorder profile if turned on')

    args=parser.parse_args()
    st=time.time()

    if args.geom:
        T_list=np.geomspace(args.Tmin,args.Tmax,args.Tnum)
    else:
        T_list=np.linspace(args.Tmin,args.Tmax,args.Tnum)
            
    inputs=[(protocol,T,args.sigma,args.sigma_t,args.noise_type,args.scaling,args.atol,args.rtol,args.savedisorder,ensemble_index) for protocol in args.protocol for T in T_list for ensemble_index in range(args.ensemble_size)]

    with MPIPoolExecutor() as executor:
        rs=list(executor.map(wrapper,inputs))
    # rs=list(map(wrapper,inputs))
    
    parity,error,disorder=zip(*rs)
    parity=np.array(parity).reshape((len(args.protocol),T_list.shape[0],args.ensemble_size))
    error=np.array(error).reshape((len(args.protocol),T_list.shape[0],args.ensemble_size))
    disorder=np.array(disorder).reshape((len(args.protocol),T_list.shape[0],args.ensemble_size))
    parity_ensemble={}
    error_ensemble={}
    disorder_ensemble={}

    for protocol, parity_0, error_0, disorder_0 in zip(args.protocol,parity,error,disorder):
        parity_ensemble[protocol]=parity_0
        error_ensemble[protocol]=error_0
        disorder_ensemble[protocol]=disorder_0

    fn = 'sigma{}_sigma_t{}_ensemble{}_scaling{}_{}_{}.pickle'.format(args.sigma,args.sigma_t,args.ensemble_size,args.scaling,args.protocol,args.noise_type)

    with open(fn,'wb') as f:
        pickle.dump([parity_ensemble,error_ensemble,args,disorder_ensemble],f)

    print(time.time()-st)


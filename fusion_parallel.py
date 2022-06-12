from Fusion import *
import argparse
from mpi4py.futures import MPIPoolExecutor
import pickle


def wrapper(inputs):
    protocol,T,sigma,sigma_t,noise_type=inputs
    fusion=Fusion(E=np.array([1]*4),t=np.array([1]*3), Delta=np.array([1]*3),T=T,noise_type=noise_type,sigma=sigma,sigma_t=sigma_t)
    parity,error=fusion.solve(protocol=protocol,method='LSODA',atol=1e-10,rtol=1e-10)
    return parity,error

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    # parser.add_argument('-protocol',type=str,help='A or B')
    # parser.add_argument('-T',type=float,help='time for one operation')
    parser.add_argument('-sigma',type=float,help='sigma for the strength of fluctuation')
    parser.add_argument('-sigma_t',type=float,help='sigma_t for the time correlation')
    parser.add_argument('-noise_type',type=str,help='noise type: E/t/Delta')
    parser.add_argument('-Tmin',default=1.,type=float,help='T min')
    parser.add_argument('-Tmax',default=200.,type=float,help='T max')
    parser.add_argument('-Tnum',default=50,type=int,help='T num')
    parser.add_argument('-ensemble_size',default=2,type=int,help='ensemble size')

    args=parser.parse_args()
    
    T_list=np.geomspace(args.Tmin,args.Tmax,args.Tnum)

    # for index, T in enumerate(T_list):
    #     for ensemble_index in range(args.ensemble_size):
    #         inputs=(protocol,T,sigma,sigma_t,noise_type)
            
    inputs=[(protocol,T,args.sigma,args.sigma_t,args.noise_type) for protocol in 'AB' for T in T_list for _  in range(args.ensemble_size)]

    with MPIPoolExecutor() as executor:
        rs=list(executor.map(wrapper,inputs))
    
    parity,error=zip(*rs)
    parity=parity.reshape((2,T_list.shape[0],args.ensemble_size))
    error=error.reshape((2,T_list.shape[0],args.ensemble_size))

    parity_ensemble['A']=parity[0]
    parity_ensemble['B']=parity[1]
    error_ensmeble['A']=error[0]
    error_ensmeble['B']=error[1]


    fn = 'sigma{:.1f}_sigma_t{:.1f}_{}'.format(args.sigma,args.sigma_t,args.noise_type)

    with open(fn,'wb') as f:
        pickle.dump(f,[parity_ensemble,error_ensmeble,args])


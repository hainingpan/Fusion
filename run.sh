# mpirun -np 32 python -m mpi4py.futures fusion_parallel.py -sigma=0 -sigma_t=1 -noise_type=E -Tnum=50 -ensemble_size=1

# mpirun -np 32 python -m mpi4py.futures fusion_parallel.py -sigma=0.0001 -sigma_t=1 -noise_type=E -Tnum=50 -ensemble_size=500
mpirun -np 32 python -m mpi4py.futures fusion_parallel.py -sigma=0.001 -sigma_t=1 -noise_type=E -Tnum=50 -ensemble_size=500
mpirun -np 32 python -m mpi4py.futures fusion_parallel.py -sigma=0.01 -sigma_t=1 -noise_type=E -Tnum=50 -ensemble_size=500
mpirun -np 32 python -m mpi4py.futures fusion_parallel.py -sigma=0.1 -sigma_t=1 -noise_type=E -Tnum=50 -ensemble_size=500
mpirun -np 32 python -m mpi4py.futures fusion_parallel.py -sigma=1 -sigma_t=1 -noise_type=E -Tnum=50 -ensemble_size=500

# mpirun -np 32 python -m mpi4py.futures fusion_parallel.py -sigma=0.0001 -sigma_t=1 -noise_type=t -Tnum=50 -ensemble_size=500
mpirun -np 32 python -m mpi4py.futures fusion_parallel.py -sigma=0.001 -sigma_t=1 -noise_type=t -Tnum=50 -ensemble_size=500
mpirun -np 32 python -m mpi4py.futures fusion_parallel.py -sigma=0.01 -sigma_t=1 -noise_type=t -Tnum=50 -ensemble_size=500
mpirun -np 32 python -m mpi4py.futures fusion_parallel.py -sigma=0.1 -sigma_t=1 -noise_type=t -Tnum=50 -ensemble_size=500
mpirun -np 32 python -m mpi4py.futures fusion_parallel.py -sigma=1 -sigma_t=1 -noise_type=t -Tnum=50 -ensemble_size=500

# mpirun -np 32 python -m mpi4py.futures fusion_parallel.py -sigma=0.0001 -sigma_t=1 -noise_type=Delta -Tnum=50 -ensemble_size=500
mpirun -np 32 python -m mpi4py.futures fusion_parallel.py -sigma=0.001 -sigma_t=1 -noise_type=Delta -Tnum=50 -ensemble_size=500
mpirun -np 32 python -m mpi4py.futures fusion_parallel.py -sigma=0.01 -sigma_t=1 -noise_type=Delta -Tnum=50 -ensemble_size=500
mpirun -np 32 python -m mpi4py.futures fusion_parallel.py -sigma=0.1 -sigma_t=1 -noise_type=Delta -Tnum=50 -ensemble_size=500
mpirun -np 32 python -m mpi4py.futures fusion_parallel.py -sigma=1 -sigma_t=1 -noise_type=Delta -Tnum=50 -ensemble_size=500
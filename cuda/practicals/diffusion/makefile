FLAGS=-arch=sm_60 -std=c++11 -O2 -lineinfo -I../include -DWITH_G2G

LINK=-lcublas -lcuda

BENCHMARKS= diffusion2d diffusion2d_mpi

all : $(BENCHMARKS)

#----------------------------------------
# diffusion2D benchmarks
#----------------------------------------
diffusion2d : diffusion2d.cu
	nvcc $(FLAGS) diffusion2d.cu $(LINK) -o diffusion2d

# an aditional linking stage required for MPI
diffusion2d_mpi : diffusion2d_mpi.cu
	nvcc -c ${FLAGS} diffusion2d_mpi.cu -I$(CRAY_MPICH2_DIR)/include
	CC diffusion2d_mpi.o -o diffusion2d_mpi -lcuda -lcudart

clean :
	rm -f ${BENCHMARKS}
	rm -f *.o
	rm -f output.*

NVCC=nvcc
#GCC=g++
EXE=cuising
SRC=main.cu loadSpins.cc
FLAGS=-O3 -use_fast_math #--ptxas-options=-v 
LIBS=-lcurand

#Location of helper_cuda.h
HOST=$(shell hostname)
ifeq ($(HOST), spicable.lassp.cornell.edu)
INC=-I/Developer/NVIDIA/CUDA-7.0/samples/common/inc -I. 
endif
ifeq ($(HOST), dain)
INC=-I/opt/cuda5.5/samples/common/inc -I.
endif
ifeq ($(HOST), jection)
INC= -I$(CUDA_HOME)/samples/common/inc -I.
endif

GPUOBJS=main.o sampling.o
#OBJS=loadSpins.o

default: main

main: $(GPUOBJS) 
	$(NVCC) $(FLAGS) -o $(EXE) $(OBJS) $(GPUOBJS) $(INC) $(LIBS)

main.o: main.cu
	$(NVCC) $(FLAGS) -c main.cu $(INC) $(LIBS)

sampling.o: sampling.cu
	$(NVCC) $(FLAGS) -c sampling.cu $(INC) $(LIBS)

.PHONY: clean
clean:
	rm -rf $(EXE) *.o


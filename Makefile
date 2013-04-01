# A simple hand-made makefile for a package including applications
# built from Fortran 90 sources, taking into account the usual
# dependency cases.

# This makefile works with the GNU make command, the one find on
# GNU/Linux systems and often called gmake on non-GNU systems, if you
# are using an old style make command, please see the file
# Makefile_oldstyle provided with the package.

# ======================================================================
# Let's start with the declarations
# ======================================================================

# Tested compilers, g95, gfortran, ifort
# The compiler

CXX = nvcc

# flags for debugging or for maximum performance, comment as necessary
#Gfortran compile flags
CXXFLAGS = -g -pg
LD_FLAGS = -Xcompiler -fopenmp  --compiler-bindir /opt/gcc-4.6.3/bin
PROGRAMS = hermes

# "make" builds all
all: $(PROGRAMS)

OBJS = 	mesh_funcs.o \
	preprocessing.o\
	cuda_query.o \
	cudapreprocessing.o

hermes: main.o ${OBJS}
	$(CXX) $(LD_FLAGS) -o hermes main.o ${OBJS}  

main.o: main.cpp 
	$(CXX) $(CXXFLAGS) -c main.cpp ${OBJS} $(LD_FLAGS) 

cudapreprocessing.o: cudapreprocessing.cu
	$(CXX) $(CXXFLAGS) -c cudapreprocessing.cu $(LD_FLAGS) 	
	
preprocessing.o: preprocessing.cpp cudapreprocessing.o
	$(CXX) $(CXXFLAGS) -c preprocessing.cpp $(LD_FLAGS) 

mesh_funcs.o: mesh_funcs.cpp
	$(CXX) $(CXXFLAGS) -c mesh_funcs.cpp $(LD_FLAGS) 

cuda_query.o: cuda_query.cpp
	$(CXX) $(CXXFLAGS) -c cuda_query.cpp $(LD_FLAGS) 
	
clean:
	rm -f ${OBJS} hermes


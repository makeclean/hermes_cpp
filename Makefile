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
LD_FLAGS = -lcurand -lcudart
PROGRAMS = hermes

# "make" builds all
all: $(PROGRAMS)

OBJS = 	main.o \
	mesh_funcs.o

hermes: ${OBJS}
	$(CXX) $(LD_FLAGS) -o hermes ${OBJS}  

main.o: main.cpp 
	$(CXX) $(CXXFLAGS) -c main.cpp mesh_funcs.o $(LD_FLAGS) 

mesh_funcs.o: mesh_funcs.cpp
	$(CXX) $(CXXFLAGS) -c mesh_funcs.cpp $(LD_FLAGS) 

clean:
	rm -f ${OBJS}


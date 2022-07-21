
SUNDIALS_INC=/home/ian/projects/sundials-6/install/include
SUNDIALS_LIB=/home/ian/projects/sundials-6/install/lib

SUNFLAGS=-I$(SUNDIALS_INC) -L$(SUNDIALS_LIB) -Wl,-rpath=$(SUNDIALS_LIB) 
SUN_LINK_FLAGS = -lsundials_kinsol -lsundials_nvecserial -lsundials_sunlinsoldense

#EIGENFLAGS=-I/usr/include/mkl
# EIG_LINK_FLAGS=-Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
CXX=g++
CXXFLAGS=-std=c++20 -g -O0 $(SUNFLAGS) $(EIGENFLAGS)
#CXXFLAGS=-std=c++20 -O2 -march=native -fopenmp $(SUNFLAGS) $(EIGENFLAGS)

LINK_FLAGS= $(SUN_LINK_FLAGS) 
all: PsiSolver

HammersteinSolver: HammersteinSolver.cpp HammersteinEquation.hpp CollocationApprox.hpp GaussNodes.hpp GeneralizedHammersteinEq.hpp
	$(CXX) -o HammersteinSolver $(CXXFLAGS) HammersteinSolver.cpp $(LINK_FLAGS)

HammersteinTests: HammersteinTests.cpp HammersteinEquation.hpp CollocationApprox.hpp GaussNodes.hpp GeneralizedHammersteinEq.hpp
	$(CXX) -o HammersteinTests $(CXXFLAGS) HammersteinTests.cpp $(LINK_FLAGS)


PsiSolver: PsiSolver.cpp HammersteinEquation.hpp CollocationApprox.hpp GaussNodes.hpp GreensFunction.hpp GreensFunction.cpp
	$(CXX) -o PsiSolver $(CXXFLAGS) PsiSolver.cpp GreensFunction.cpp $(LINK_FLAGS)

clean:
	rm -f HammersteinSolver PsiSolver;

.PHONY: clean all

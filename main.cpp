/* Main program for the c++ version of hermes-gpu */
#include <iostream>
#include <vector> 

#include <time.h>       // for timing the routine

#include "MeshData.hpp"
#include "PreProcess.hpp"
#include "CudaData.hpp"

int main ( int argc, char *argv[] )
{
  int errorcode;
  std::string meshfile;
  int num_vols; // number of volumes
  std::vector<tet_struct> tet_data;
  std::vector<node_struct> node_data;

  time_t time_before,time_after; // for timing routines

  // parse the command line arguments
  for ( int argnum = 0 ; argnum <= argc-1 ; argnum++ )
    {
      if ( argv[argnum] == std::string("--mesh") )
	{
	  meshfile = std::string(argv[argnum+1]); // convert to std::string
	  // return refs to node data and tet data
	  std::cout << "Reading mesh ..." << std::endl;
	  errorcode = LoadMeshFile(meshfile,node_data,tet_data,num_vols); // load the file pointed to
	  if ( errorcode != 0 )
	    {
	      return 1;
	      // clean up since we have failed to read the mesh
	    }

	  PrintMeshVtk(meshfile,node_data,tet_data); // dump the geometry mesh to file
	}

    }
 
  errorcode = CudaQuery(); // query the cuda devices
  if ( errorcode == 0 )
    {
      std::cerr << "There were no CUDA capable devices found" << std::endl;
      std::cerr << "Consider running in MPI or OMP mode" << std::endl;
    }
  time(&time_before); 
  std::cout << "Preprocess" << std::endl;
  errorcode = PreProcess(node_data,tet_data); // Pre process the mesh data, renumber etc
  time(&time_after); 
  double seconds = difftime(time_after,time_before); //get the time in seconds
  std::cout << "serial version took " << seconds << " s" << std::endl;

  time(&time_before); 
  std::cout << "CudaPreprocess" << std::endl;
  errorcode = CudaPreProcess(node_data,tet_data); // GPU Pre process the mesh data, renumber 
  time(&time_after); 
  seconds = difftime(time_after,time_before); //get the time in seconds
  std::cout << "gpu version took " << seconds << " s" << std::endl;

  return 0;
}



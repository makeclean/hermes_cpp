/* Main program for the c++ version of hermes-gpu */
#include <iostream>
#include <vector> 

#include "MeshData.hpp"

int main ( int argc, char *argv[] )
{
  int errorcode;
  std::string meshfile;
  int num_vols; // number of volumes
  std::vector<tet_struct> tet_data;
  std::vector<node_struct> node_data;

  // parse the command line arguments
  for ( int argnum = 0 ; argnum <= argc-1 ; argnum++ )
    {
      std::cout << argv[argnum] << std::endl;
      if ( argv[argnum] == std::string("--mesh") )
	{
	  meshfile = std::string(argv[argnum+1]); // convert to std::string
	  // return refs to node data and tet data
	  errorcode = LoadMeshFile(meshfile,node_data,tet_data,num_vols); // load the file pointed to
	  if ( errorcode != 0 )
	    {
	      // clean up since we have failed to read the mesh
	    }

	  std::cout << tet_data.size() << " num vls = " << num_vols << std::endl;

	}
    }

  return 0;
}



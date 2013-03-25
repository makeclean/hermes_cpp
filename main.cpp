/* Main program for the c++ version of hermes-gpu */
#include <iostream>
#include <vector> 

#include "MeshData.hpp"

int main ( int argc, char *argv[] )
{
  int errorcode;
  std::string meshfile;
  std::vector<tet_struct> tet_data;
  std::vector<node_struct> node_data;

  // parse the command line arguments
  for ( int argnum = 0 ; argnum <= argc ; argnum++ )
    {
      std::cout << argv[argnum] << std::endl;
      if ( argv[argnum] == std::string("--mesh") )
	{
	  meshfile = std::string(argv[argnum+1]); // convert to std::string
	  // return refs to node data and tet data
	  errorcode = LoadMeshFile(meshfile,node_data,tet_data); // load the file pointed to
	}
    }

  return 0;
}



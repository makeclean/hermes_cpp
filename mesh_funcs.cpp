#include <iostream>
#include <string>
#include <fstream>
#include <vector>

#include "MeshData.hpp"

using namespace std;

/*
  Function to load the abaqus mesh file pointed to by meshfile
  On failure it will return the value of 1 to main
 */

int LoadMeshFile(std::string meshfile, std::vector<node_struct> &node_data
		 ,std::vector<tet_struct> &tet_data)
{
  std::ifstream mesh_file; // the abaqus mesh
  std::string line_read; // the line being read from file

  std::cout << "Reading abaqus mesh " << meshfile << std::endl;
  
  mesh_file.open(meshfile.c_str(), ios::in); // open the mesh file
  
  if( mesh_file.is_open() )
    {
      // do stuff
    }
  else
    {
      std::cout << "Failed to open mesh file, " << meshfile << std::endl;
      return 1;
    }


  return 0;
}

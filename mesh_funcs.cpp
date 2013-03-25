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

  node_struct tmp_node_data; // temporary copy of the current lines node data

  std::cout << "Reading abaqus mesh " << meshfile << std::endl;
  
  mesh_file.open(meshfile.c_str(), ios::in); // open the mesh file
  
  if( mesh_file.is_open() ) // while we can read data
    {
      int set_allnodes = 0 ; // when file is reading the line NSET=ALLNODES
      while( getline(mesh_file,line_read) ) //take the current line and decide what do 
	{
	  if ( set_allnodes == 1 )
	    {
	      tmp_node_data = NodeLineToNodeData(line_read); // convert the line being read into node struct
	    }

	  if (std::string::npos != line_read.find("NSET=ALLNODES"))
	    {
	      std::cout << line_read << std::endl;
	      set_allnodes = 1;
	    }

	}
    }
  else
    {
      std::cout << "Failed to open mesh file, " << meshfile << std::endl;
      return 1;
    }


  return 0;
}

/*
 *
 */
node_struct NodeLineToNodeData(std::string line_read)
{
  node_struct mesh_node_structure;

  std::cout << "ey up" << std::endl;

  return mesh_node_structure;
}


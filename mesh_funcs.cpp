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
		 ,std::vector<tet_struct> &tet_data, int &num_vols)
{
  std::ifstream mesh_file; // the abaqus mesh
  std::string line_read; // the line being read from file

  node_struct tmp_node_data; // temporary copy of the current lines node data

  std::cout << "Reading abaqus mesh " << meshfile << std::endl;
  
  mesh_file.open(meshfile.c_str(), ios::in); // open the mesh file
  
  if( mesh_file.is_open() ) // while we can read data
    {
      int set_allnodes = 0 ; // when file is reading the line NSET=ALLNODES
      int skip_line = 0; // when to skip a line or not
      int set_tet_data =0 ; // when we have reached tets 
      while( getline(mesh_file,line_read) ) //take the current line and decide what do 
	{
	  if ( set_allnodes == 1 )
	    {
	      tmp_node_data = NodeLineToNodeData(line_read); // convert the line being read into node struct
	      node_data.push_back(tmp_node_data); // push the node information to node data
	    }

	  if (std::string::npos != line_read.find("NSET=ALLNODES")) // look for node data
	    {
	      std::cout << line_read << std::endl;
	      set_allnodes = 1;
	    }

	  if (std::string::npos != line_read.find("*ELEMENT, TYPE")) // look for volume entity information
	    {
	      num_vols += 1; // increment num_vols by 1
	      skip_line = 1; // since we have read element,type we now need to skip reading a line
	      set_tet_data = 1;
	    }

	  if ( set_tet_data == 1 && skip_line == 0 )
	    {
	      tmp_tet_data = TetLinetoTetData(line_read);
	      std::cout << line_read << std::endl;
	    }  

	  skip_line = 0;
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
 *   function to take a string and convert the string to node_structure format
 */
node_struct NodeLineToNodeData(std::string line_read)
{
  node_struct mesh_node_structure; // struct of node data
  std::string tmp_copy;

  unsigned pos = line_read.find(","); // find comma in the string
  mesh_node_structure.node_number = atoi(line_read.substr(0,pos).c_str()); //assign node id

  tmp_copy = line_read.substr(pos+1,line_read.length()); // copy from after comma to end of the string
  line_read = tmp_copy; // copy to original

  pos = line_read.find(",");
  mesh_node_structure.x_coord = atof(line_read.substr(0,pos).c_str()); //assign x coord

  tmp_copy = line_read.substr(pos+1,line_read.length()); // copy from after comma to end of the string
  line_read = tmp_copy; // copy to original

  pos = line_read.find(",");
  mesh_node_structure.y_coord = atof(line_read.substr(0,pos).c_str()); //assign y coord
		     
  tmp_copy = line_read.substr(pos+1,line_read.length()); // copy from after comma to end of the string
  line_read = tmp_copy; // copy to original
  mesh_node_structure.z_coord = atof(line_read.substr(0,pos).c_str()); //assign y coord

  return mesh_node_structure;
}

tet_struct TetLineToTetData(std::string line_read)
{
  

  return mesh_tet_structure;
}


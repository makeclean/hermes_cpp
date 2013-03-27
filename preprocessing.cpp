#include <iostream>
#include <string>
#include <vector>

#include "MeshData.hpp"
#include "PreProcess.hpp"

/*
 * 	Function to preprocess the mesh to reorder the element from 1 to N, 
 * 	to calculate the volume of the tets 
 * 	and the adjacancy information
 */

int PreProcess(std::vector<node_struct> &node_data, std::vector<tet_struct> &tet_data)
{
  int errorcode; // return error code from functions
  
  std::cout << "Performing preprocessing" << std::endl;
  
  errorcode = ReNumberNodes(node_data); // renumber the nodes from 1 to N
  
  errorcode = ReNumberTets(tet_data); // renumber the nodes from 1 to M
  
  std::cout << "Calculating volumes " << std::endl;
  
  return 0;
}

/*
 * 	renumber the nodes from 0 to N-1
 */
int ReNumberNodes(std::vector<node_struct> &node_data)
{
    std::vector<node_struct>::iterator node_it; //iterator through node_data
    for ( node_it = node_data.begin() ; node_it != node_data.end() ; ++node_it )
      {
	std::cout << (node_it->node_number) << std::endl;
      }
      
    return 0;
}

/*
 * 	renumber the tets from 0 to N-1
 */
int ReNumberTets(std::vector<tet_struct> &tet_data)
{
    std::vector<tet_struct>::iterator tet_it; //iterator through tet_data
    for ( tet_it = tet_data.begin() ; tet_it != tet_data.end() ; ++tet_it )
      {
	std::cout << (tet_it->tet_num) << std::endl;
      }
      
    return 0;
}

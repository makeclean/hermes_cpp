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
  std::vector<node_struct> copy_original_node; // copy of node data
  
  int errorcode; // return error code from functions
  
  std::cout << "Performing preprocessing" << std::endl;
  
  errorcode = ReNumberNodes(node_data, copy_original_node); // renumber the nodes from 1 to N
  
  errorcode = ReNumberTets(copy_original_node,tet_data); // renumber the nodes from 1 to M
  
  std::cout << "Calculating volumes " << std::endl;
  
  return 0;
}

/*
 * 	renumber the nodes from 0 to N-1, i.e. makes the node list self referential 
 */
int ReNumberNodes(std::vector<node_struct> &node_data, std::vector<node_struct> &original_nodes)
{
    int num_node = 0; // node num index
    std::vector<node_struct>::iterator node_it; //iterator through node_data
    node_struct copy_of_nodes; // structure template
    
    for ( node_it = node_data.begin() ; node_it != node_data.end() ; ++node_it )
      {
	copy_of_nodes.node_number = node_data[num_node].node_number; // copy the old node number
	original_nodes.push_back(copy_of_nodes); // push the original data to vector
	node_data[num_node].node_number = num_node; 
	num_node++;
      }
      
    return 0;
}

/*
 * 	renumber the tets from 0 to N-1
 */
int ReNumberTets(std::vector<node_struct> node_data, std::vector<tet_struct> &tet_data)
{
    std::vector<tet_struct>::iterator tet_it; //iterator through tet_data
    std::vector<node_struct>::iterator node_it;
    
    int node_num = 0;
    int tet_num = 0;
    
    for ( node_it = node_data.begin() ; node_it != node_data.end() ; ++node_it )
      {
	for ( tet_it = tet_data.begin() ; tet_it != tet_data.end() ; ++tet_it )
	  {
	    if ( ( tet_it->link1 ) == ( node_it->node_number ) )
	      tet_it->link1 = node_num; 
	    if ( ( tet_it->link2 ) == ( node_it->node_number ) )
	      tet_it->link2 = node_num; 
	    if ( ( tet_it->link3 ) == ( node_it->node_number ) )
	      tet_it->link3 = node_num; 
	    if ( ( tet_it->link4 ) == ( node_it->node_number ) )
	      tet_it->link4 = node_num;	    
	  }
	  
	node_num++;
      } 
     
     for ( tet_it = tet_data.begin() ; tet_it != tet_data.end() ; ++tet_it )
       {
	     tet_it->tet_num = tet_num;
	     tet_num++;
	 
       }
     
    return 0;
}

#include <iostream>
#include <string>
#include <vector>
#include <math.h>

#include <omp.h>

#include "MeshData.hpp"
#include "PreProcess.hpp"

int WrapCalculateTetVolume(std::vector<node_struct> node_data, std::vector<tet_struct> tet_data);

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
  
  errorcode = WrapCalculateTetVolume(node_data,tet_data); // call gpu wrapper
  
  std::cout << "Calculating volumes " << std::endl;
  
  errorcode = CalculateTetVolume(node_data,tet_data); // calculate the tet vols
  
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

int CalculateTetVolume(std::vector<node_struct> node_data, std::vector<tet_struct> &tet_data)
{
   int node1,node2,node3,node4; //id numbers of the nodes
   int tet_it;
   vector a,b,c,d;
   
   double total = 0.0;
  
   #pragma omp parallel for
   for ( tet_it = 0 ; tet_it <= tet_data.capacity()-1 ; tet_it++ )
      {
	#pragma omp critical
	{
	  int th_id = omp_get_thread_num();  
	  std::cout << th_id << std::endl;
	} 

       
	/*
	#pragma omp master
	{
	  int nthreads = omp_get_num_threads();
	  std::cout << "There are " << nthreads << " threads" << '\n';
	}

	*/
	
	 node1 = tet_data[tet_it].link1;
	 node2 = tet_data[tet_it].link2;
	 node3 = tet_data[tet_it].link3;
	 node4 = tet_data[tet_it].link4;
	
	 
	 a.i = node_data[node1].x_coord-node_data[node4].x_coord;
	 a.j = node_data[node1].y_coord-node_data[node4].y_coord;
	 a.k = node_data[node1].z_coord-node_data[node4].z_coord;

	 b.i = node_data[node2].x_coord-node_data[node4].x_coord;
	 b.j = node_data[node2].y_coord-node_data[node4].y_coord;
	 b.k = node_data[node2].z_coord-node_data[node4].z_coord;
	 
	 c.i = node_data[node3].x_coord-node_data[node4].x_coord;
	 c.j = node_data[node3].y_coord-node_data[node4].y_coord;
	 c.k = node_data[node3].z_coord-node_data[node4].z_coord;
	 
	 d.i = (a.j*b.k) - (a.k*b.j); //cross product
	 d.j = (a.k*b.i) - (a.i*b.k); //cross product
	 d.k = (a.i*b.j) - (a.j*b.i); //cross product
	 
	 tet_data[tet_it].volume = fabs((c.i*d.i)+(c.j*d.j)+(c.k*d.k))/6.0; //dot product
	 total += tet_data[tet_it].volume;
	 //std::cout << (tet_it->volume) << std::endl;
	}
      
      std::cout << total << std::endl;
   return 0;
}


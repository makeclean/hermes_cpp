#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <stdlib.h>

#include "MeshData.hpp"
#include "PreProcess.hpp"

/*
 *  PreProcess takes the input data and reorders the mesh such that 
 */
int PreProcess(std::vector<node_struct> &node_data, std::vector<tet_struct> &tet_data)
{
  std::vector<int> NewNodeNumbers; // vector for new node numbers
  int counter; // count of tets

  // reorder the data 
  std::vector<node_struct>::const_iterator node_it;

  std::cout << "Preprocessing ..." << std::endl;
  for ( node_it = node_data.begin() ; node_it != node_data.end() ; ++node_it)
    {
      counter++; // add one to count
      NewNodeNumbers.push_back(counter); // add one to the array
    }


  if ( node_data.size() != NewNodeNumbers.size() ) 
    {
      std::cerr << "New node numbers is not equal to length of original" << std::endl;
      exit(9);
    }

  // loop over the node set 
  for ( int i = 1 ; i <= node_data.size() ; i++ )
    {
      std::vector<tet_struct>::iterator tet_it;
      // replace the list from M to L from 1 to L-M, 
      for ( tet_it = tet_data.begin() ; tet_it != tet_data.end() ; ++tet_it )
	{
	  if ( (tet_it->link1) == node_data[i].node_number )
	    {
	      (*tet_it).link1 = i;
	    }
	  if ( (tet_it->link2) == node_data[i].node_number )
	    {
	      (*tet_it).link2 = i;
	    }
	  if ( (tet_it->link3) == node_data[i].node_number )
	    {
	      (*tet_it).link3 = i;
	    }
	  if ( (tet_it->link4) == node_data[i].node_number )
	    {
	      (*tet_it).link4 = i;
	    }
	}
    }

  // now have re-ordered list
  
  /*
  std::vector<tet_struct>::iterator tet_it;
  // replace the list from M to L from 1 to L-M, 
  for ( tet_it = tet_data.begin() ; tet_it != tet_data.end() ; ++tet_it )
    {
      std::cout << (*tet_it).link1 << " " << (*tet_it).link2 << " "
		<< (*tet_it).link3 << " " << (*tet_it).link4 << std::endl;
    }
  */

  int i = CalculateVolume(node_data, tet_data);

  return 0;
}

/*
 * Function to calculate the volume of each tetrahedron
 */
int CalculateVolume(std::vector<node_struct> &node_data, std::vector<tet_struct> &tet_data)
{
  std::vector<tet_struct>::iterator tet_it;
  int nodes[4];
  double vertex_positions[4][3];
  double total_volume = 0.0;

  std::cout << "Calculating volumes ..." << std::endl;

  // loop over the tet data
  for ( tet_it = tet_data.begin() ; tet_it != tet_data.end() ; ++tet_it )
    {
      nodes[0]=(*tet_it).link1;
      nodes[1]=(*tet_it).link2;
      nodes[2]=(*tet_it).link3; 
      nodes[3]=(*tet_it).link4;
     
      for ( int i = 0 ; i <= 3 ; i++ ) 
	{
	  vertex_positions[i][0]=node_data[nodes[i]].x_coord;
	  vertex_positions[i][1]=node_data[nodes[i]].y_coord;
	  vertex_positions[i][2]=node_data[nodes[i]].z_coord;      
	}

      double a[3],b[3],c[3],d[3],e[3],f[3],g[3]; // arrays for vector subtraction
      // copy data from vertex arrays to temp vars
      for ( int i = 0 ; i <= 2 ; i++)
	{
	  a[i]=vertex_positions[0][i];
	  b[i]=vertex_positions[1][i];
	  c[i]=vertex_positions[2][i];
	  d[i]=vertex_positions[3][i];
	}

      // vector subtraction
      for ( int i = 0 ; i <= 2 ; i++ )
	{
	  e[i]=a[i]-d[i];
	  f[i]=b[i]-d[i];
	  g[i]=c[i]-d[i];
	}

      double h[3];
      // cross product
      h[0] = (f[1]*g[2]) - (f[2]*g[1]);
      h[1] = (f[2]*g[0]) - (f[0]*g[2]);
      h[2] = (f[0]*g[1]) - (f[1]*g[0]);

      double total = 0.0;
      //dot product
      for ( int i = 0 ; i <= 2 ; i++ )
	{
	  total += (e[i]*h[i]);
	}

      (*tet_it).volume = abs(total/6.);
      total_volume += (*tet_it).volume;
    }

  std::cout << "total volume = " << total_volume << std::endl;

  return 1;
}

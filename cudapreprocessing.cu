#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <stdlib.h>

#include <new>

#include "MeshData.hpp"

using namespace std;


/*
 * 	cuda version of calculate tet volume
 */
__global__ void CudaCalculateTetVolume(int num_tets)
{
  int index = threadIdx.x + (blockIdx.x*blockDim.x);
  if ( index < num_tets )
    {
      
    }

    syncthreads();

}


/*
 *  Wrapper around CudaCalculateTetVolume to reform the C++ struct data to standard C forms
 *  and copies the data to the gpu memory
 *  Reforms std::vector<node_struct> into a normal array of ints
 *  Reforms std::vector<tet_struct> into a normal array of floats
 */
int WrapCalculateTetVolume(std::vector<node_struct> node_data, std::vector<tet_struct> tet_data)
{
       
    int num_tets  = tet_data.capacity();  // number of elements
    int num_nodes = node_data.capacity(); // number of nodes
    
    std::cout << "in gpu code" << std::endl;
    
    // Allocate vectors in device memory
    double* tet_volumes;
    cudaMalloc((void**) &tet_volumes, num_tets*sizeof(double)); // allocate storage for tet_vols, should be the size of num_vols

    // create arrays for x coords
    double* x_coords_dev;
    double  x_coords[num_nodes];
    
    cudaMalloc((void**) &x_coords_dev, num_nodes*sizeof(double)); // allocate storage x coords, should be size of num_nodes
    
    // create arrays for y coords
    double* y_coords_dev;
    double  y_coords[num_nodes];
    
    cudaMalloc((void**) &y_coords_dev, num_nodes*sizeof(double)); // allocate storage x coords, should be size of num_nodes
    
     // create arrays for z coords
    double* z_coords_dev;
    double  z_coords[num_nodes];
    
    cudaMalloc((void**) &z_coords_dev, num_nodes*sizeof(double)); // allocate storage x coords, should be size of num_nodes

    // copy struct data to 'normal' arrays
    for ( int i = 0 ; i < num_nodes ; i++ )
      {
	x_coords[i] = node_data[i].x_coord;
	y_coords[i] = node_data[i].y_coord;
	z_coords[i] = node_data[i].z_coord;
      }
    
    cudaMemcpy(x_coords_dev,x_coords, num_nodes,cudaMemcpyHostToDevice); // copy x coords to gpu
    cudaMemcpy(y_coords_dev,y_coords, num_nodes,cudaMemcpyHostToDevice); // copy y coords to gpu
    cudaMemcpy(z_coords_dev,z_coords, num_nodes,cudaMemcpyHostToDevice); // copy z coords to gpu

    
    int threads_per_block = 256;
    int blocks_per_grid = 12;
    CudaCalculateTetVolume <<<blocks_per_grid, threads_per_block>>> (num_tets);

    delete[] x_coords;
    delete[] y_coords;
    delete[] z_coords;
    
    cudaFree(tet_volumes); // deallocate tet_volumes array
    return 0;
}
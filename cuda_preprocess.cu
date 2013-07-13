#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "MeshData.hpp"

// Device code
__global__ void cudaPreProc (int *DEV_nodenumbers, double *DEV_xcoords,
			     double *DEV_ycoords, double *DEV_zcoords,
			     int *DEV_tetnumbers, int *DEV_connectionsa,int *DEV_connectionsb,
			     int *DEV_connectionsc,int *DEV_connectionsd,
			     int DEV_numnodes, int DEV_numtets, double *DEV_volumes)
{
  const unsigned int tid = threadIdx.x + (blockDim.x*blockIdx.x);
  // tet index is tid
  int connect[4];
  // get the connect data
  connect[0]=DEV_connectionsa[tid];
  connect[1]=DEV_connectionsb[tid];
  connect[2]=DEV_connectionsc[tid];
  connect[3]=DEV_connectionsd[tid];

  // now determine the node positions
  double positions[4][3];
  for ( int i = 0 ; i <= 3 ; i++)
    {
      positions[i][0]=DEV_xcoords[connect[i]];
      positions[i][1]=DEV_ycoords[connect[i]];
      positions[i][2]=DEV_zcoords[connect[i]];
    }

  double a[3],b[3],c[3],d[3],e[3],f[3],g[3]; // arrays for vector subtraction
  // copy data from vertex arrays to temp vars
  for ( int i = 0 ; i <= 2 ; i++)
    {
      a[i]=positions[0][i];
      b[i]=positions[1][i];
      c[i]=positions[2][i];
      d[i]=positions[3][i];
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

  DEV_volumes[tid] = abs(total)/6.0;
      
  //  DEV_nodenumbers[tid] = DEV_numnodes-tid;
}

/*
__global__ void CudaPreproc(std::vector<node_struct> node_data, std::vector<tet_struct> tet_data)
{
   const unsigned int tid = threadIdx.x;
}
*/

int CudaPreProcess(std::vector<node_struct> &node_data, 
                   std::vector<tet_struct> &tet_data)
{
	
  int devicecount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&devicecount);
  int dev;
  for (dev = 0; dev < devicecount; ++dev)
    cudaSetDevice(dev);
  int num_nodes=node_data.size(); // get length of node data

  /*
  int *DEV_test; 
  int test_int = 5000;

  std::cout << num_nodes << std::endl;
  cudaMalloc((void**)&DEV_test, test_int*sizeof(int)); 
  std::cout << "ok " << std::endl;
  cudaFree(DEV_test);
  cudaMalloc((void**)&DEV_test, num_nodes*sizeof(int)); 
  return 1;
  */
  
  int *node_numbers;
  double *x_coord,*y_coord,*z_coord;
  node_numbers = new int[num_nodes];

  x_coord = new double[num_nodes];
  y_coord = new double[num_nodes];
  z_coord = new double[num_nodes];

  // allocate node data
  for ( int i = 0 ; i <= num_nodes-1 ; i++ )
    {
      node_numbers[i] = node_data[i].node_number;
      //      std::cout << node_numbers[i] << std::endl;
      x_coord[i] = node_data[i].x_coord;
      y_coord[i] = node_data[i].y_coord;
      z_coord[i] = node_data[i].z_coord;
    }

  // Copy the data to device
  int *DEV_nodenumbers; 
  std::cout << num_nodes << std::endl;

  cudaMalloc((void**)&DEV_nodenumbers, num_nodes*sizeof(int)); 
  std::cout << num_nodes << std::endl;
  cudaMemcpy(DEV_nodenumbers,node_numbers,num_nodes*sizeof(int),cudaMemcpyHostToDevice);

  double *DEV_xcoords;
  cudaMalloc((void**) &DEV_xcoords, num_nodes*sizeof(double));
  cudaMemcpy(DEV_xcoords,x_coord,num_nodes*sizeof(double),cudaMemcpyHostToDevice);

  double *DEV_ycoords;
  cudaMalloc((void**) &DEV_ycoords, num_nodes*sizeof(double));
  cudaMemcpy(DEV_ycoords,y_coord,num_nodes*sizeof(double),cudaMemcpyHostToDevice);

  double *DEV_zcoords;
  cudaMalloc((void**) &DEV_zcoords, num_nodes*sizeof(double));
  cudaMemcpy(DEV_zcoords,z_coord,num_nodes*sizeof(double),cudaMemcpyHostToDevice);

  // all node data copied

  // allocate tet data
  int num_tets=tet_data.size(); // get length of node data
  
  int *tet_numbers;
  tet_numbers = new int[num_tets];
  int connectionsa[num_tets],connectionsb[num_tets];
  int connectionsc[num_tets],connectionsd[num_tets];

  for (int i = 0 ; i <= num_tets-1 ; i++ )
    {
      tet_numbers[i] = tet_data[i].tet_num;
      connectionsa[i] = tet_data[i].link1;
      connectionsb[i] = tet_data[i].link2;
      connectionsc[i] = tet_data[i].link3;
      connectionsd[i] = tet_data[i].link4;
    }

  // allocate tet data on gpu

  // Copy the data to device
 int *DEV_tetnumbers; 
 cudaMalloc((void**) &DEV_tetnumbers, num_tets*sizeof(int)); 
 cudaMemcpy(DEV_tetnumbers,tet_numbers,num_tets*sizeof(int),cudaMemcpyHostToDevice);

 // copy connection data
 int *DEV_connectionsa, *DEV_connectionsb, *DEV_connectionsc,*DEV_connectionsd;
 cudaMalloc((void**) &DEV_connectionsa,num_tets*sizeof(int));
 cudaMalloc((void**) &DEV_connectionsb,num_tets*sizeof(int));
 cudaMalloc((void**) &DEV_connectionsc,num_tets*sizeof(int));
 cudaMalloc((void**) &DEV_connectionsd,num_tets*sizeof(int));

 cudaMemcpy(DEV_connectionsa,connectionsa,num_tets*sizeof(int),cudaMemcpyHostToDevice);
 cudaMemcpy(DEV_connectionsb,connectionsb,num_tets*sizeof(int),cudaMemcpyHostToDevice);
 cudaMemcpy(DEV_connectionsc,connectionsc,num_tets*sizeof(int),cudaMemcpyHostToDevice);
 cudaMemcpy(DEV_connectionsd,connectionsd,num_tets*sizeof(int),cudaMemcpyHostToDevice);
 
 std::cout << "calling the kernel" << std::endl;
 int numth = (num_tets+255)/256;

 double *DEV_volumes;
 cudaMalloc((void**) &DEV_volumes,num_tets*sizeof(double));

 // run the preproc kernel
 cudaPreProc<<<numth,256>>>(DEV_nodenumbers, DEV_xcoords, DEV_ycoords,DEV_zcoords,
			    DEV_tetnumbers, DEV_connectionsa,DEV_connectionsb,
			    DEV_connectionsc,DEV_connectionsd,
			    num_nodes, num_tets, DEV_volumes);
  
 double tet_volumes[num_tets];

 //  cudaMemcpy(node_numbers,DEV_nodenumbers,num_nodes*sizeof(int),cudaMemcpyDeviceToHost);
 cudaMemcpy(tet_volumes,DEV_volumes,num_tets*sizeof(double),cudaMemcpyDeviceToHost);
 std::cout << "run done" << std::endl;
 return 1;
 for( int i = 0 ; i <= num_tets-1 ; i++ )
   {
     //std::cout << node_data[i].node_number << " " << node_numbers[i] << std::endl;
     std::cout << tet_volumes[i] << std::endl;
   }
 std::cout << "size of data" << node_data.size() << std::endl;

 return 1;
}



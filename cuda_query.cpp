#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "CudaData.hpp"

int CudaQuery()
{
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  std::cout << "There are " << deviceCount << " devices" << std::endl;
  return 0;
}

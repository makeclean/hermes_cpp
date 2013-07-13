#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "CudaData.hpp"

/*
 *  Queries and prints list of CUDA capable devices, returns 0 if there were none
 */
int CudaQuery()
{
  int devicecount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&devicecount);
  std::cout << "+===============================================================+" << std::endl;
  std::cout << "There are " << devicecount << " device(s)" << std::endl;
  std::cout << "+===============================================================+" << std::endl;

  int dev, driverVersion = 0, runtimeVersion = 0;
  for (dev = 0; dev < devicecount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);	

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

        char msg[256];
        sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
        printf("%s", msg);

        printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

  std::cout << "+===============================================================+" << std::endl;

    }
  cudaSetDevice(devicecount-1);
  return devicecount;
}

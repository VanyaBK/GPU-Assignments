#include<stdio.h>
#include<cuda.h>
#include "kernels.h"

__global__ void per_row_kernel(int *in,int N){
  int row = threadIdx.x * blockDim.y + threadIdx.y + blockIdx.x * blockDim.x * blockDim.y;
  if(row<N)
  {
    for(int i=0;i<N;i++)
    {
      if(i>row)
      {
        in[N*row + i] = in[i*N + row];
        in[i*N + row] = 0;
      }
    }
  }
}

__global__ void per_element_kernel(int *in, int N){
  long int ele = (blockIdx.x*gridDim.y+blockIdx.y)*(gridDim.z*blockDim.x)+(blockIdx.z*blockDim.x+threadIdx.x);
  if(ele < N*N-1)
  {
    int x = ele/N;
    int y = ele%N;
    if(ele > x*N+x)
    {
      in[ele] = in[y*N+x];
      in[y*N+x] = 0;
    }
  }	
}

__global__ void per_element_kernel_2D(int *in, int N){
  long int ele = (blockIdx.x*gridDim.y+blockIdx.y)*(blockDim.x*blockDim.y)+(threadIdx.x*blockDim.y+threadIdx.y);
  if(ele < N*N-1)
  {
    int x = ele/N;
    int y = ele%N;
    if(ele > x*N+x)
    {
      in[ele] = in[y*N+x];
      in[y*N+x] = 0;
    }
  }
}

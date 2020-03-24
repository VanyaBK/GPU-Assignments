#include<stdio.h>
#include<cuda.h>

__global__ void sumRandC(int* A, int* B, int m, int n, int p, int q, int k)
{
  int id=blockIdx.x*blockDim.x + threadIdx.x,idx;
  if(id<((m*n)/k))
  {
  for(int i=0;i<k;i++)
  {
    idx = id+i*((m*n)/k);
    B[idx+(idx/n)] = A[idx];
    atomicAdd(&B[(((idx/n)+1)*n)+(idx/n)],A[idx]);  // Adds elements to the row end
    atomicAdd(&B[(m*n)+m+(idx%n)],A[idx]); // Adds elements to the column end
    if(idx==0)
      B[p*q-1] = INT_MAX;
  }
  }
}
__global__ void findMIn( int* A, int* B, int m, int n, int p, int q, int k)
{
  int id=blockIdx.x*blockDim.x + threadIdx.x,idx;
  if(id<((m*n)/k))
  {
  for(int i=0;i<k;i++)
  {
    idx=id+i*((m*n)/k);
    atomicMin(&B[p*q-1],B[(((idx/n)+1)*n)+(idx/n)]); // Checks minimum of row end elements
    atomicMin(&B[p*q-1],B[(m*n)+m+(idx%n)]); // Checks minimum of column end elements
  }
  }
}
__global__ void updateMin( int* A, int* B, int m, int n, int p, int q, int k)
{
  int id=blockIdx.x*blockDim.x + threadIdx.x,idx;
  if(id<((m*n)/k))
  {
  for(int i=0;i<k;i++)
  {
    idx = id+i*((m*n)/k)+((id+i*((m*n)/k))/n);
    if(idx%q!=n && idx/q!=m)
    { 
      atomicAdd(&B[idx],B[p*q-1]); // Adds minimum to all the elements not in the last row and column
    }
  }
  }
}
int main() 
{ 
  int M,N,k;
  scanf( "%d %d %d", &M,&N,&k);
  int *matrix,*matrix1, *hmatrix,*h1matrix;
  cudaMalloc(&matrix, (M) * (N) * sizeof(int));
  cudaMalloc(&matrix1, (M+1) * (N+1) * sizeof(int));
  hmatrix = (int *)malloc(M * N * sizeof(int));
  h1matrix = (int *)malloc((M+1) * (N+1) * sizeof(int));
  for (int ii = 0; ii < M; ++ii) 
  {
    for (int jj = 0; jj < N; ++jj) 
    {
      scanf("%d",&hmatrix[ii*N+jj]);
    }
  }
  cudaMemcpy(matrix, hmatrix, M * N * sizeof(int), cudaMemcpyHostToDevice);
  sumRandC<<<ceil((float)(M*N)/(k*1024)),1024>>>(matrix,matrix1,M,N,M+1,N+1,k);
  findMIn<<<ceil((float)(M*N)/(k*1024)),1024>>>(matrix,matrix1,M,N,M+1,N+1,k);
  updateMin<<<ceil((float)(M*N)/(k*1024)),1024>>>(matrix,matrix1,M,N,M+1,N+1,k);
  cudaDeviceSynchronize();
  cudaMemcpy(h1matrix, matrix1, (M+1) * (N+1) * sizeof(int), cudaMemcpyDeviceToHost);
  for (int ii = 0; ii < M+1; ++ii) 
  {
    for (int jj = 0; jj < N+1; ++jj) 
    {
      printf("%d ",h1matrix[ii*(N+1)+jj]);
    } 
    printf("\n");
  }
  return 0;
}

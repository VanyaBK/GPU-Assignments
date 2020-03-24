#include<stdio.h>
#include<cuda.h>

__global__ void update(int *matrix,int *query,int *row,int *row_ele,int *no_query,int *prev_array,int n,int m,int q)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	if(id<(m*q))
	{

	int query_no = id/m;
	int row_no = id%m;
	if(matrix[row_no*n+row[query_no]-1]==row_ele[query_no])   // Updating if the row element matches
	{	
		for(int i=0;i<no_query[query_no];i++)
		{
			if(query[prev_array[query_no]+i*3+2]==0)
			{
				atomicSub(&matrix[row_no*n+query[prev_array[query_no]+i*3]-1] , query[prev_array[query_no]+i*3+1]);
			}
			else
			{
				atomicAdd(&matrix[row_no*n+query[prev_array[query_no]+i*3]-1] , query[prev_array[query_no]+i*3+1]);
			}	
		}
	}
	}	
}

int main(int argc,char **argv)
{
  int M,N,q;
  FILE *fpi,*fpo;
  fpi=fopen(argv[1],"r");
  fpo=fopen(argv[2],"w");
  fscanf(fpi,"%d %d", &M,&N);
  int *matrix, *hmatrix;
  char character;int *row,*row_ele,*drow,*drow_ele;
  int *query,*no_query,*prev_array,*dquery,*dno_query,*dprev_array;
  cudaMalloc(&matrix, (M) * (N) * sizeof(int));
  hmatrix = (int *)malloc(M * N * sizeof(int));
  for (int ii = 0; ii < M; ++ii) 
  {
    for (int jj = 0; jj < N; ++jj) 
    {
      fscanf(fpi,"%d",&hmatrix[ii*N+jj]);
    }
  }
  cudaMemcpy(matrix, hmatrix, M * N * sizeof(int), cudaMemcpyHostToDevice);
  fscanf(fpi,"%d", &q);
  cudaMalloc(&dquery, 90 * q * sizeof(int));
  cudaMalloc(&drow, q * sizeof(int));
  cudaMalloc(&drow_ele, q * sizeof(int));
  cudaMalloc(&dno_query, q * sizeof(int));
  cudaMalloc(&dprev_array, q * sizeof(int));
  query = (int *)malloc(90 * q * sizeof(int));
  row = (int *)malloc(q * sizeof(int));
  row_ele = (int *)malloc(q * sizeof(int));
  no_query = (int *)malloc(q * sizeof(int));
  prev_array = (int *)malloc(q * sizeof(int));
  int prev=0;char c1[50];

  // Parsing Queries
  for (int i = 0; i < q; i++)
  {
  	fscanf(fpi,"%[^U]s",c1);
  	fscanf(fpi,"%c",&character);
  	fscanf(fpi," %c",&character);
  	fscanf(fpi,"%d %d %d",&row[i],&row_ele[i],&no_query[i]);
  	for(int j=0;j<no_query[i];j++)
  	{
  		fscanf(fpi," %c %d %d %c",&character,&query[prev+(j*3)],&query[prev+(j*3)+1],&character);
  		if(character=='+')
  			query[prev+(j*3)+2]=1;
  		else
  			query[prev+(j*3)+2]=0;
  	}
  	prev_array[i]=prev;
  	prev += no_query[i]*3;
  }
  cudaMemcpy(dquery, query, 90 * q * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(drow, row,  q * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(drow_ele, row_ele, q * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dno_query, no_query, q * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dprev_array, prev_array, q * sizeof(int), cudaMemcpyHostToDevice);
  update<<<3000,1024>>>(matrix,dquery,drow,drow_ele,dno_query,dprev_array,N,M,q);
  cudaMemcpy(hmatrix, matrix, M * N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int ii = 0; ii < M; ++ii) 
  {
    for (int jj = 0; jj < N; ++jj) 
    {
    	if(jj==N-1)
    		fprintf(fpo,"%d ",hmatrix[ii*N+jj]);
    	else
      		fprintf(fpo,"%d ",hmatrix[ii*N+jj]);
    }
    fprintf(fpo,"\n");
  }
  return 0;
}
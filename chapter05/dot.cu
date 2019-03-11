#include<iostream>
#define N 33*256
#define threadPerBlock 256
#define blockPerGrid 32
__global__ void dot(float *a,float *b,float *c)
{
	__shared__ float cache[threadPerBlock];
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int cacheIndex = threadIdx.x;
	float temp =0;
	while(tid < N)
	{
		temp += a[tid]*b[tid];
		tid += blockDim.x * gridDim.x;
	}
	cache[cacheIndex] = temp;
	__syncthreads();

	int i=blockDim.x/2;
	while(i!=0)
	{
		if(cacheIndex < i)
				cache[cacheIndex] += cache[cacheIndex+i];
		__syncthreads();
		i /= 2;
	}
	if(cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}	
int main()
{
	float *a=(float *)malloc(sizeof(float)*N);
	float *b=(float *)malloc(sizeof(float)*N);
	float *c=(float *)malloc(sizeof(float)*blockPerGrid);

	float *dev_a;
	float *dev_b;
	float *dev_c;

	cudaMalloc((void**)&dev_a,sizeof(float)*N);
	cudaMalloc((void**)&dev_b,sizeof(float)*N);
	cudaMalloc((void**)&dev_c,sizeof(float)*blockPerGrid);

	int i;
	for(i=0;i<N;i++)
		a[i]=b[i]=1;

	cudaMemcpy(dev_a,a,sizeof(float)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b,sizeof(float)*N,cudaMemcpyHostToDevice);
	dot<<<blockPerGrid,threadPerBlock>>>(dev_a,dev_b,dev_c);
	cudaMemcpy(c,dev_c,sizeof(float)*blockPerGrid,cudaMemcpyDeviceToHost);

	float sum=0;
	for(i=0;i<blockPerGrid;i++)
			sum+=c[i];

	printf("sum is :%f \n",sum);

	free(a);
	free(b);
	free(c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	
}

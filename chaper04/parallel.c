#include<iostream>
using namespace std;
#define N 10
__global__ void add(int *a,int *b,int *c)
{
	int tid = blockIdx.x;
	if(tid < N)
	{
		c[tid] = a[tid] + b[tid];
	}
}

int main()
{
	int a[N],b[N],c[N];
	int *dev_a;
	int *dev_b;
	int *dev_c;
	int i;

	cudaMalloc((void **)&dev_a,sizeof(int)*N);
	cudaMalloc((void **)&dev_b,sizeof(int)*N);
	cudaMalloc((void **)&dev_c,sizeof(int)*N);

	for(i=0;i<N;i++)
	{
		a[i]=i;
		b[i]=i;
	}
	cudaMemcpy(dev_a,a,sizeof(int)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b,sizeof(int)*N,cudaMemcpyHostToDevice);

	add<<<N,1>>>(dev_a,dev_b,dev_c);

	cudaMemcpy(c,dev_c,sizeof(int)*N,cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	for(i=0;i<N;i++)
	{
		printf("%d + %d = %d\n",a[i],b[i],c[i]);
	}

	return 0;
}

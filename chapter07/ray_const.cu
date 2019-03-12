#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#define INF 2e10f
#define rnd(x) (x*rand()/RAND_MAX)
#define SPHERES 20
#define DIM 1024
using namespace cv;
struct Sphere
{
	float x,y,z;
	float radius;
	float r,g,b;
	__device__ float hit(float ox,float oy,float *n)
	{
		float dx = ox - x;
		float dy = oy - y;
		if(dx*dx + dy*dy <= radius*radius)
		{
			float dz = sqrt(radius*radius - dx*dx - dy*dy);
			*n = dz/sqrtf(radius*radius);
			return dz+z;
		}
		return -INF;
	}
};
__constant__ Sphere s[SPHERES];
__global__ void kernel(unsigned char *dev_mat)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = x + y*blockDim.x*gridDim.x;
	float ox = (x-DIM/2);
	float oy = (y-DIM/2);

	int i;
	float maxd=-INF;
	float d;
	float scale;
	float r=0;
	float g=0;
	float b=0;
	float n;
	for(i=0;i<SPHERES;i++)
	{
		d=s[i].hit(ox,oy,&n);
		if(d > maxd)
		{
			scale = n;
			r = s[i].r*scale;
			g = s[i].g*scale;
			b = s[i].b*scale;
			maxd = d;
	//		printf("r:%f g:%f b:%f\n",r,g,b);
		}
		
	}


	dev_mat[4*offset+0]=(int)(r*255);
	dev_mat[4*offset+1]=(int)(g*255);
	dev_mat[4*offset+2]=(int)(b*255);
	dev_mat[4*offset+3]=255;
}
int main()
{
	Mat mat(DIM,DIM,CV_8UC4);
	int size = mat.cols*mat.rows*mat.elemSize();
	unsigned char *dev_mat;
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	cudaMalloc((void **)&dev_mat,size);

	struct Sphere s_temp[SPHERES];
	int i;
	for(i=0;i<SPHERES;i++)
	{
		s_temp[i].r = rnd(1.0f);
		s_temp[i].g = rnd(1.0f);
		s_temp[i].b = rnd(1.0f);
		s_temp[i].x = rnd(1000.0f) -500;
		s_temp[i].y = rnd(1000.0f) -500;
		s_temp[i].z = rnd(1000.0f) -500;
		s_temp[i].radius = rnd(100.0f) + 20;
	}
	cudaMemcpyToSymbol(s,s_temp,sizeof(Sphere)*SPHERES);

	dim3 grids(DIM/16,DIM/16);
	dim3 threads(16,16);



	kernel<<<grids,threads>>>(dev_mat);
	
	cudaMemcpy(mat.ptr(),dev_mat,size,cudaMemcpyDeviceToHost);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("Time to generate %f ms\n",elapsedTime);

	namedWindow("display",CV_WINDOW_AUTOSIZE);
	imshow("display",mat);
	cvWaitKey(0);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(dev_mat);
	cudaFree(s);
	return 0;
}

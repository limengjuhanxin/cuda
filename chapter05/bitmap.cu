#include<iostream>
#include<math.h>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
#define DIM 1024
#define PI 3.1415926535897932f
__global__ void bitmap(unsigned char *ptr)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	__shared__ float shared[16][16];

	int offset = x + y*gridDim.x*blockDim.x;
	const float period = 128.0f;


	shared[threadIdx.x][threadIdx.y]=255*(sin(x*2.0f*PI/period)+1.0f)*(sin(y*2.0f*PI/period)+1.0f)/4.0f;

	__syncthreads();
	ptr[offset*4+0] =0;
	ptr[offset*4+1]=shared[15-threadIdx.x][15-threadIdx.y];
	ptr[offset*4+2]=0;
	ptr[offset*4+3]=255;
}
int main()
{
	Mat mat(DIM,DIM,CV_8UC4);
	int size = mat.cols*mat.rows*mat.elemSize();
	unsigned char *dev_mat;
	cudaMalloc((void **)&dev_mat,size);
	dim3 blocks(DIM/16,DIM/16);
	dim3 threads(16,16);
	bitmap<<<blocks,threads>>>(dev_mat);
	cudaMemcpy(mat.ptr(),dev_mat,size,cudaMemcpyDeviceToHost);
	namedWindow("display",CV_WINDOW_AUTOSIZE);
	imshow("display",mat);
	cudaFree(dev_mat);
	cvWaitKey(0);
	return 0;
}

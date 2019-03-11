#include<iostream>
#include<opencv2/opencv.hpp>
#define DIM 1000
using namespace std;
using namespace cv;
struct cuComplex{
	float r;
	float i;
	__device__ cuComplex(float r,float i):r(r),i(i){}
	__device__ float magnitude2(){return r*r + i*i;}
	__device__ cuComplex operator *(const cuComplex &a){return cuComplex(r*a.r -i*a.i, i*a.r+r*a.i);}
	__device__ cuComplex operator +(const cuComplex &a){return cuComplex(r+a.r,i+a.i);}
	__device__ void print(){ printf("r:%f i:%f \n",r,i);}
};
__device__ int julia(int x,int y)
{
	float scale = 1.5;
	float jx = scale*(float)(DIM/2-x)/(DIM/2);
	float jy = scale*(float)(DIM/2-y)/(DIM/2);

	cuComplex c(-0.8,0.156);
	cuComplex a(jx,jy);

	int i;
	for(i=0;i<200;i++)
	{
		a = a*a + c;
		if(a.magnitude2()>1000)
			return 0;	
	}
	return 1;
}
__global__ void kernel(unsigned char *ptr)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;//gridDim内置常数，用来保存每一维的大小，这里gridDim(DIM,DIM)
	int juliaValue = julia(x,y);
	ptr[offset*4 + 0] = 255*juliaValue;
	ptr[offset*4 + 1] = 0;
	ptr[offset*4 + 2] = 0;
	ptr[offset*4 + 3] = 255;
}
int main()
{
	Mat mat(DIM,DIM,CV_8UC4);
	unsigned char *dev_mat;
	int size = mat.rows*mat.cols*mat.elemSize();
	cudaMalloc((void **)&dev_mat,size);
	dim3 grid(DIM,DIM);//dim3:运行时需要一个三维的dim3的值，自动把第三位设定为1
	kernel<<<grid,1>>>(dev_mat);
	if(cudaMemcpy(mat.ptr(0,0),dev_mat,size,cudaMemcpyDeviceToHost)!=cudaSuccess)
	{
		printf("error\n");
	}
	namedWindow("display",CV_WINDOW_AUTOSIZE);
	imshow("display",mat);
	printf("done\n");
	cvWaitKey(0);
	cudaFree(dev_mat);
	return 0;
}

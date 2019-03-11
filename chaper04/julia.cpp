#include<iostream>
#include<opencv2/opencv.hpp>
#define DIM 1000
using namespace std;
using namespace cv;
struct cuComplex
{
	float r;
	float i;
	cuComplex(float r,float i):r(r),i(i){}
	float magnitude2(){return r*r + i*i;}
	cuComplex operator *(const cuComplex &a){return cuComplex( r*a.r - i*a.i,i*a.r+r*a.i);}
	cuComplex operator +(const cuComplex &a){return cuComplex(r+a.r,i+a.i);}
};
int julia(int x, int y)
{
	float scale=1.5;
	float jx = scale*(float)(DIM/2 - x)/(DIM/2);
	float jy = scale*(float)(DIM/2 - y)/(DIM/2);

	cuComplex a(jx,jy);
	cuComplex c(-0.80,0.156);

	int i;
	for(i=0; i<200;i++)
	{
		a= a*a + c;
		if(a.magnitude2() >1000)
			return 0;
	}
	return 1;
}
int kernel(unsigned char *ptr)
{
	int x,y;
	for(y=0;y<DIM;y++)
	{
		for(x=0;x<DIM;x++)
		{
			int offset = y*DIM + x;
			int juliaValue = julia(x,y);
			ptr[offset*4 + 0] = 255*juliaValue;
			ptr[offset*4 + 1] = 0;
			ptr[offset*4 + 2] = 0;
			ptr[offset*4 + 3] = 255;
		}
	}
}
int main()
{
	cv::Mat mat(DIM,DIM,CV_8UC4);
	unsigned char *ptr = mat.ptr(0);
	kernel(ptr);
	namedWindow("display",CV_WINDOW_AUTOSIZE);
	imshow("display",mat);
	cvWaitKey();
	return 0;
}

#include "cpu_anim.h"
#define DIM 1024
#define PI 3.1415926535897932f
struct DataBlock
{
	CPUAnimBitmap *bitmap;
	unsigned char * dev_bitmap;
};
__global__ void kernel(unsigned char *ptr,int ticks)
{
	int x=threadIdx.x + blockIdx.x*blockDim.x;
	int y=threadIdx.y + blockIdx.y*blockDim.y;
	int offset = x + y*blockDim.x*gridDim.x;

	float fx = x - DIM/2;
	float fy = y - DIM/2;
	float d = sqrtf(fx*fx + fy*fy);

	unsigned char grey = (unsigned char)(128.0f + 127.0f *cos(d/10.0f - ticks/7.0f) /(d/10.0f + 1.0f));

	ptr[offset*4+0]=grey;
	ptr[offset*4+1]=grey;
	ptr[offset*4+2]=grey;
	ptr[offset*4+3]=255;
}
void generate_frame(DataBlock *d,int ticks)
{
	dim3 grids(DIM/16,DIM/16);
	dim3 threads(16,16);
	kernel<<<grids,threads>>>(d->dev_bitmap,ticks);
	cudaMemcpy(d->bitmap->get_ptr(),d->dev_bitmap,d->bitmap->image_size(),cudaMemcpyDeviceToHost);
}
void clean_up(DataBlock *d)
{
	cudaFree(d->dev_bitmap);
	delete d->bitmap;
}
int main()
{
	DataBlock data;
	data.bitmap = new CPUAnimBitmap(DIM,DIM,&data);
	cudaMalloc((void **)&data.dev_bitmap,data.bitmap->image_size());
	data.bitmap->anim_and_exit((void (*)(void *,int))generate_frame,(void (*)(void *))clean_up);
}

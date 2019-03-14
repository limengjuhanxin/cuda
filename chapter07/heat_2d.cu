#include "cpu_anim.h"
#include "book.h"


#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED   0.25f

texture<float,2> texIn;
texture<float,2> texOut;
texture<float,2> texConst;

struct DataBlock
{
	float * dev_inSrc;
	float * dev_outSrc;
	float *dev_constSrc;
	unsigned char *output_bitmap;
	
	CPUAnimBitmap *bitmap;

	cudaEvent_t start,stop;
	float total_time;
	int frames;
};
__global__ void copy_to_const(float * in)
{

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = x + y*blockDim.x*gridDim.x;

	float value = tex2D(texConst,x,y);
	if(value!=0)
	{
		in[offset] =value;
	}
}
__global__ void blend_kernel(float * out,float dstOut)
{

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = x + y*blockDim.x*gridDim.x;

	float t,b,c,l,r;
	if(dstOut)
	{
		t = tex2D(texIn,x,y-1);
		b = tex2D(texIn,x,y+1);
		l = tex2D(texIn,x-1,y);
		r = tex2D(texIn,x+1,y);
		c = tex2D(texIn,x,y);
	}
	else
	{

		t = tex2D(texOut,x,y-1);
		b = tex2D(texOut,x,y+1);
		l = tex2D(texOut,x-1,y);
		r = tex2D(texOut,x+1,y);
		c = tex2D(texOut,x,y);
	}

	out[offset] = c + SPEED*(t+b+r+l-4*c);

}

void anim_gpu(DataBlock *d,int ticks)
{
	dim3 blocks(DIM/16,DIM/16);
	dim3 threads(16,16);

	int i;
	bool dstOut=true;
	float *in;
	float *out;

	cudaEventRecord(d->start,0);
	for(i=0;i<90;i++)
	{
		if(dstOut==true)
		{
			in = d->dev_inSrc;
			out = d->dev_outSrc;
		}
		else
		{
			in = d->dev_outSrc;
			out = d->dev_inSrc;
		}
		
		copy_to_const<<<blocks,threads>>>(in);
		blend_kernel<<<blocks,threads>>>(out,dstOut);

		dstOut = !dstOut;
	}

	float_to_color<<<blocks,threads>>>(d->output_bitmap,d->dev_inSrc);
	cudaMemcpy(d->bitmap->get_ptr(),d->output_bitmap,d->bitmap->image_size(),cudaMemcpyDeviceToHost);

	cudaEventRecord(d->stop,0);
	cudaEventSynchronize(d->stop);
	float elasped_time;
	cudaEventElapsedTime(&elasped_time,d->start,d->stop);

	d->total_time += elasped_time;
	d->frames++;

	printf("ave time:%f \n",d->total_time/d->frames);
}
void anim_exit(DataBlock *d)
{
	cudaEventDestroy(d->start);
	cudaEventDestroy(d->stop);

	cudaUnbindTexture(texIn);
	cudaUnbindTexture(texOut);
	cudaUnbindTexture(texConst);


	cudaFree(d->dev_inSrc);
	cudaFree(d->dev_outSrc);
	cudaFree(d->dev_constSrc);
	cudaFree(d->output_bitmap);
}
int main()
{

	DataBlock data;
	CPUAnimBitmap bitmap(DIM,DIM,&data);
	data.bitmap = &bitmap;
	data.total_time = 0;
	data.frames = 0;

	cudaEventCreate(&data.start);
	cudaEventCreate(&data.stop);


	int imageSize = bitmap.image_size();
	cudaMalloc((void**)&data.dev_inSrc,imageSize);
	cudaMalloc((void**)&data.dev_outSrc,imageSize);
	cudaMalloc((void**)&data.dev_constSrc,imageSize);
	cudaMalloc((void**)&data.output_bitmap,imageSize);


	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL,texIn,data.dev_inSrc,desc,DIM,DIM,DIM*sizeof(float));
	cudaBindTexture2D(NULL,texOut,data.dev_outSrc,desc,DIM,DIM,DIM*sizeof(float));
	cudaBindTexture2D(NULL,texConst,data.dev_constSrc,desc,DIM,DIM,DIM*sizeof(float));



    // initialize the constant data
  	float *temp = (float*)malloc( imageSize );
  	for (int i=0; i<DIM*DIM; i++) {
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if ((x>300) && (x<600) && (y>310) && (y<601))
            temp[i] = MAX_TEMP;
	 }
	temp[DIM*100+100] = (MAX_TEMP + MIN_TEMP)/2;
	temp[DIM*700+100] = MIN_TEMP;
	temp[DIM*300+300] = MIN_TEMP;
	temp[DIM*200+700] = MIN_TEMP;
	for (int y=800; y<900; y++) {
	for (int x=400; x<500; x++) {
		temp[x+y*DIM] = MIN_TEMP;
		}
	   }
	 cudaMemcpy( data.dev_constSrc, temp,imageSize,cudaMemcpyHostToDevice);    

	    // initialize the input data
	 for (int y=800; y<DIM; y++) {
		for (int x=0; x<200; x++) {
		    temp[x+y*DIM] = MAX_TEMP;
		}
	    }
	cudaMemcpy( data.dev_inSrc, temp,imageSize,cudaMemcpyHostToDevice );
	free( temp );







	bitmap.anim_and_exit((void(*)(void *,int))anim_gpu,(void (*)(void *))anim_exit);


}

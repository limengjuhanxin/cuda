#include"book.h"
#include "cpu_anim.h"
#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED   0.25f
struct DataBlock
{
	float *dev_inSrc;
	float *dev_outSrc;
	float *dev_constSrc;
	CPUAnimBitmap *bitmap;
	unsigned char *output_bitmap;
	cudaEvent_t start,stop;
	float total_time;
	int frames;
};
texture<float> texIn;
texture<float> texOut;
texture<float> texConst;


__global__ void copy_to_const(float *iptr)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = x +y*blockDim.x*gridDim.x;
	
	float c = tex1Dfetch(texConst,offset);
	if(c!=0) 
		iptr[offset]=c;
	
}

__global__ void blend_kernel(float* dst,bool dstOut)
{

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = x +y*blockDim.x*gridDim.x;

	int top = offset - DIM;
	if(y==0) top += DIM;
	int bottom = offset + DIM;
	if(y==DIM-1) bottom -= DIM;

	int left = offset-1;
	if(x==0) left+=1;
	int right = offset +1;
	if(x==DIM-1) right-=1;


	float t,l,c,r,b;
	if(dstOut)
	{
		t = tex1Dfetch(texIn,top);
		l = tex1Dfetch(texIn,left);
		c = tex1Dfetch(texIn,offset);
		r = tex1Dfetch(texIn,right);
		b = tex1Dfetch(texIn,bottom);
	}
	else
	{

		t = tex1Dfetch(texOut,top);
		l = tex1Dfetch(texOut,left);
		c = tex1Dfetch(texOut,offset);
		r = tex1Dfetch(texOut,right);
		b = tex1Dfetch(texOut,bottom);
	}

	dst[offset] = c + SPEED*(t+l+r+b-4*c);
}
void anim_gpu(DataBlock *data,int ticks)
{
	
	int i;
	dim3 blocks(DIM/16,DIM/16);
	dim3 threads(16,16);
	cudaEventRecord(data->start,0);

	bool dstOut = true;
	float *in;
	float *out;
	for(i=0;i<90;i++)
	{
		if(dstOut)
		{
			in = data->dev_inSrc;
			out = data->dev_outSrc;
		}
		else
		{
			in = data->dev_outSrc;;
			out = data->dev_inSrc;;
		}
		copy_to_const<<<blocks,threads>>>(in);
		blend_kernel<<<blocks,threads>>>(out,dstOut);
		dstOut = !dstOut;
	}
	float_to_color<<<blocks,threads>>>(data->output_bitmap,data->dev_inSrc);
	cudaMemcpy(data->bitmap->get_ptr(),data->output_bitmap,data->bitmap->image_size(),cudaMemcpyDeviceToHost);

	cudaEventRecord(data->stop,0);
	cudaEventSynchronize(data->stop);
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time,data->start,data->stop);
	++data->frames;
	data->total_time += elapsed_time;
	printf("average time:%f\n",data->total_time/data->frames);
}
void anim_exit(DataBlock *data)
{

	cudaUnbindTexture(texIn);
	cudaUnbindTexture(texOut);
	cudaUnbindTexture(texConst);

	cudaFree(data->output_bitmap);
	cudaFree(data->dev_inSrc);
	cudaFree(data->dev_outSrc);
	cudaFree(data->dev_constSrc);


	cudaEventDestroy(data->start);
	cudaEventDestroy(data->stop);
}
int main()
{
	DataBlock data;
	CPUAnimBitmap bitmap(DIM,DIM,&data);
	data.bitmap = &bitmap;
	data.frames = 0;
	data.total_time = 0;

	cudaEventCreate(&data.start);
	cudaEventCreate(&data.stop);

	int image_size=bitmap.image_size();

	cudaMalloc((void **)&data.output_bitmap,image_size);
	cudaMalloc((void **)&data.dev_inSrc,image_size);
	cudaMalloc((void **)&data.dev_outSrc,image_size);
	cudaMalloc((void **)&data.dev_constSrc,image_size);


	cudaBindTexture(NULL,texIn,data.dev_inSrc,image_size);
	cudaBindTexture(NULL,texOut,data.dev_outSrc,image_size);
	cudaBindTexture(NULL,texConst,data.dev_constSrc,image_size);



 	float *temp = (float*)malloc( image_size );
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
    HANDLE_ERROR( cudaMemcpy( data.dev_constSrc, temp,
                              image_size,
                              cudaMemcpyHostToDevice ) );    

    // initialize the input data
    for (int y=800; y<DIM; y++) {
        for (int x=0; x<200; x++) {
            temp[x+y*DIM] = MAX_TEMP;
        }
    }
    HANDLE_ERROR( cudaMemcpy( data.dev_inSrc, temp,
                              image_size,
                              cudaMemcpyHostToDevice ) );
    free( temp );


//	bitmap.anim_and_exit((void (*)(void *,int))anim_gpu,(void (*)(void *)anim_exit));

	bitmap.anim_and_exit((void(*)(void *,int))anim_gpu,(void(*)(void*))anim_exit);
}

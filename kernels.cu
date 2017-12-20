

#ifndef KERNELS_CU
#define KERNELS_CU
#include <math.h>



//LSTM Kernels

typedef float (*op_func_t) (float); 
//Initial: ft = sigmoid(W_f*concatenate(h_t-1,x_t]+bf))
__device__ float sigmoid(float x){
    float retVal = 0;
    retVal = 1+exp(x);
    retVal = 1/retVal;
    return retVal;
}
__device__ float sigmoidDerivative(float x){
    float val = sigmoid(x);
    return val*(1-val);
}
/*
__global__ void applySigmoid(float* hiddenLayer){
    int index = threadIdx.x;
    hiddenLayer[index] = sigmoid(hiddenLayer[index]);
}*/
__global__ void backPropagationFirstKernel(float* inputLayer,float* outputLayer,float* outputLayerTrue,float* weights,float* deltas,float* dels,float alpha,float lrate){
    int tindex = threadIdx.x;
    int oindex = blockIdx.x;
    int windex = tindex+blockIdx.x*blockDim.x;
    float del = (outputLayer[oindex]-outputLayerTrue[oindex])*outputLayer[oindex]*(1-outputLayer[oindex]);
    deltas[windex] = (1-alpha)*lrate*del*inputLayer[tindex]+alpha*deltas[windex];
    weights[windex] = weights[windex]-deltas[windex];
}

//blocks: (# of output layers,# of hidden layers)
__global__ void backPropagationSecondKernelPart1(float* hiddenLayer,float* gammas,float* weights,float* dels,float alpha,float lrate){
    int tindex = threadIdx.x;//For each hidden node
    int oindex = blockIdx.x;//for each output node
    //int mindex = blockIdx.y;//for each hidden layer
    int windex = tindex+blockIdx.x*blockDim.x;
    //int sindex = tindex+blockIdx.y*gridDim.x*blockDim.x;
    //float gamma = 0;
    gammas[windex] = dels[oindex]*weights[windex]*hiddenLayer[tindex]*(1-hiddenLayer[tindex]);
    //float del = (outputLayer[oindex]-outputLayerTrue[oindex])*outputLayer[oindex]*(1-outputLayer[oindex]);
    //deltas[windex] = (1-alpha)*lrate*del*inputLayer[tindex]+alpha*deltas[windex];
    //weights[windex] = weights[windex]-deltas[windex];
}
/*
__global__ void backPropagationSecondKernelPart15(float* gammas,int kSize){
    int tindex = threadIdx.x;//For each hidden node
    int gindex = tindex*blockDim.x;
    for(int i = 1; i < kSize;i++){
        gammas[gindex]+=gammas[gindex+i];
    }
}*/

__global__ void backPropagationSecondKernelPart2(float* inputLayer,float* gammas,float* weights,float* deltas,float alpha,float lrate){
    int tindex = threadIdx.x;//For each upper node
    int oindex = blockIdx.x;//for each hidden node
    int windex = tindex+blockIdx.x*blockDim.x;//
    int gindex = blockIdx.x*gridDim.x;
    deltas[windex] = (1-alpha)*lrate*gammas[gindex]*inputLayer[tindex]+alpha*deltas[windex];
    weights[windex] = weights[windex]-deltas[windex];
}
__global__ void forwardPropagation(float* x,float*Y,float* W,int yWidth){
    int tindex = threadIdx.x;
    int yindex = threadIdx.x+blockIdx.x*yWidth;//yWidth different because it has to be an exponent of 2
    int windex = threadIdx.x+blockIdx.x*blockDim.x;
    Y[yindex] = x[tindex]*W[windex];
    //then perform reduction,then sigmoid
}
__global__ void sigmoidKernel(float* y){
    y[threadIdx.x] = sigmoid(y[threadIdx.x]);
}

__global__ void matrixReductionDestructive(float *g_data)
{
    extern __shared__ float sdata[];
    unsigned int tindex = (threadIdx.x);
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    sdata[tindex] = g_data[i];
    __syncthreads();
    for(unsigned int s = blockDim.x/2;s>=0;s>>=1){
        //int index = 2*s*tindex;
        if((tindex<(s+1))&&((tindex+s+1)<blockDim.x)){
            sdata[tindex]+=sdata[tindex+s+1];
        }
        __syncthreads();
    }
    if(tindex==0) g_data[blockIdx.x] = sdata[0];
}
__global__ void matrixReduction(float *g_data,float* o_data)
{
    extern __shared__ float sdata[];
    unsigned int tindex = (threadIdx.x);
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    sdata[tindex] = g_data[i];
    __syncthreads();
    for(unsigned int s = blockDim.x/2;s>=0;s>>=1){
        //int index = 2*s*tindex;
        if((tindex<(s+1))&&((tindex+s+1)<blockDim.x)){
            sdata[tindex]+=sdata[tindex+s+1];
        }
        __syncthreads();
    }
    if(tindex==0) o_data[blockIdx.x] = sdata[0];
}

#endif
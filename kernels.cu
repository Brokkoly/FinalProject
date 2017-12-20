


#include <math.h>



//LSTM Kernels


//Initial: ft = sigmoid(W_f*concatenate(h_t-1,x_t]+bf))
__device__ float sigmoid(float x){
    float retVal = 0;
    retval = 1+exp(x);
    retval = 1/retval;
    return retval;
}
__device__ float sigmoidDerivative(float x){
    float val = sigmiod(x);
    return val*(1-val);
}

__global__ void applySigmoid(float* hiddenLayer){
    int index = threadIDx.x;
    hiddenLayer[index] = sigmoid(hiddenLayer[index]);
}
__global__ void backPropagationFirstKernel(float* inputLayer,float* outputLayer,float* outputLayerTrue,float* weights,float* deltas,float* dels,float alpha,float lrate){
    int tindex = threadIDx.x;
    int oindex = blockIDx.x;
    int windex = tindex+blockIDx.x*blockDim.x;
    float del = (outputLayer[oindex]-outputLayerTrue[oindex])*outputLayer[oindex]*(1-outputLayer[oindex]);
    deltas[windex] = (1-alpha)*lrate*del*inputLayer[tindex]+alpha*deltas[windex];
    weights[windex] = weights[windex]-deltas[windex];
}

//blocks: (# of output layers,# of hidden layers)
__global__ void backPropagationSecondKernelPart1(float* hiddenLayer,float* gammas,float* weights,float* dels,float alpha,float lrate){
    int tindex = threadIDx.x;//For each hidden node
    int oindex = blockIDx.x;//for each output node
    //int mindex = blockIDx.y;//for each hidden layer
    int windex = tindex+blockIDx.x*blockDim.x;
    //int sindex = tindex+blockIDx.y*gridDim.x*blockDim.x;
    //float gamma = 0;
    gammas[gindex] = dels[oindex]*weights[windex]*h[tindex]*(1-h[tindex]);
    //float del = (outputLayer[oindex]-outputLayerTrue[oindex])*outputLayer[oindex]*(1-outputLayer[oindex]);
    //deltas[windex] = (1-alpha)*lrate*del*inputLayer[tindex]+alpha*deltas[windex];
    //weights[windex] = weights[windex]-deltas[windex];
}
/*
__global__ void backPropagationSecondKernelPart15(float* gammas,int kSize){
    int tindex = threadIDx.x;//For each hidden node
    int gindex = tindex*blockDim.x;
    for(int i = 1; i < kSize;i++){
        gammas[gindex]+=gammas[gindex+i];
    }
}*/

__global__ void backPropagationSecondKernelPart2(float* inputLayer,float* gammas,float* weights,float* deltas,float* alpha,float lrate){
    int tindex = threadIDx.x;//For each upper node
    int oindex = blockIDx.x;//for each hidden node
    int windex = tindex+blockIDx.x*blockDim.x;//
    deltas[windex] = (1-alpha)*lrate*gammas[oindex]*inputLayer[tindex]+alpha*deltas[windex];
    weights[windex] = weights[windex]-deltas[windex];
}
__global__ void forwardPropagation(float* x,float*Y,float* W,int yWidth){
    int tindex = threadIDx.x;
    int yindex = threadIDx.x+blockIDx.x*yWidth;//yWidth different because it has to be an exponent of 2
    int windex = threadIDx.x+blockIDx.x*blockDim.x;
    Y[yindex] = x[tindex]*W[windex];
    //then perform reduction,then sigmoid
}
__global__ void sigmoidKernel(float* y){
    y[threadIDx.x] = sigmoid(y[threadIDx.x]);
}

__global__ void matrixReduction(float *g_data,float *out_data)
{
    extern __shared__ float sdata[];
    unsigned int tindex = (threadIdx.x);
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    sdata[tindex] = g_data[i];
    __syncthreads();
    for(unsigned int s = blockDim.x/2;s>0;s>>=1){
        //int index = 2*s*tindex;
        if(tindex<s){
            sdata[tindex]+=sdata[tindex+s];
        }
        __syncthreads();
    }
    if(tindex==0) out_data[blockIdx.x] = sdata[0];
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


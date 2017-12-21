#ifndef KERNELS_CU
#define KERNELS_CU
#include <math.h>



//LSTM Kernels

typedef double (*op_func_t) (double); 
//Initial: ft = sigmoid(W_f*concatenate(h_t-1,x_t]+bf))
__device__ double sigmoid(double x){
    double retVal = 0;
    retVal = 1+exp(x);
    retVal = 1/retVal;
    return retVal;
}
__device__ double sigmoidDerivative(double x){
    double val = sigmoid(x);
    return val*(1-val);
}
/*
__global__ void applySigmoid(double* hiddenLayer){
    int index = threadIdx.x;
    hiddenLayer[index] = sigmoid(hiddenLayer[index]);
}*/
__global__ void backPropagationFirstKernel(double* inputLayer,double* outputLayer,double* outputLayerTrue,double* weights,double* deltas,double* dels,double alpha,double lrate){
    int tindex = threadIdx.x;
    int oindex = blockIdx.x;
    int windex = tindex+blockIdx.x*blockDim.x;
    double del = (outputLayer[oindex]-outputLayerTrue[oindex])*outputLayer[oindex]*(1-outputLayer[oindex]);
    deltas[windex] = (1-alpha)*lrate*del*inputLayer[tindex]+alpha*deltas[windex];
    weights[windex] = weights[windex]-deltas[windex];
}

//blocks: (# of output layers,# of hidden layers)
__global__ void backPropagationSecondKernelPart1(double* hiddenLayer,double* gammas,double* weights,double* dels,double alpha,double lrate){
    int tindex = threadIdx.x;//For each hidden node
    int oindex = blockIdx.x;//for each output node
    //int mindex = blockIdx.y;//for each hidden layer
    int windex = tindex+blockIdx.x*blockDim.x;
    //int sindex = tindex+blockIdx.y*gridDim.x*blockDim.x;
    //double gamma = 0;
    gammas[windex] = dels[oindex]*weights[windex]*hiddenLayer[tindex]*(1-hiddenLayer[tindex]);
    //double del = (outputLayer[oindex]-outputLayerTrue[oindex])*outputLayer[oindex]*(1-outputLayer[oindex]);
    //deltas[windex] = (1-alpha)*lrate*del*inputLayer[tindex]+alpha*deltas[windex];
    //weights[windex] = weights[windex]-deltas[windex];
}
/*
__global__ void backPropagationSecondKernelPart15(double* gammas,int kSize){
    int tindex = threadIdx.x;//For each hidden node
    int gindex = tindex*blockDim.x;
    for(int i = 1; i < kSize;i++){
        gammas[gindex]+=gammas[gindex+i];
    }
}*/

__global__ void backPropagationSecondKernelPart2(double* inputLayer,double* gammas,double* weights,double* deltas,double alpha,double lrate){
    int tindex = threadIdx.x;//For each upper node
    int oindex = blockIdx.x;//for each hidden node
    int windex = tindex+blockIdx.x*blockDim.x;//
    int gindex = blockIdx.x*gridDim.x;
    deltas[windex] = (1-alpha)*lrate*gammas[gindex]*inputLayer[tindex]+alpha*deltas[windex];
    weights[windex] = weights[windex]-deltas[windex];
}
__global__ void forwardPropagation(double* x,double*Y,double* W,int yWidth,double offset){
    int tindex = threadIdx.x;
    int yindex = threadIdx.x+blockIdx.x*yWidth;//yWidth different because it has to be an exponent of 2
    int windex = threadIdx.x+blockIdx.x*blockDim.x;
    Y[yindex] = x[tindex]*W[windex];
    if(tindex==0)Y[yindex]+=offset;
    //then perform reduction,then sigmoid
}
__global__ void sigmoidKernel(double* y){
    y[threadIdx.x] = sigmoid(y[threadIdx.x]);
}

__global__ void matrixReductionDestructive(double *g_data,int size,int biggerSize)
{
    extern __shared__ double sdata[];
    unsigned int tindex = (threadIdx.x);
    unsigned int i = blockIdx.x*size+threadIdx.x;
    sdata[tindex] = g_data[i];
    __syncthreads();
    //int temp = blockDim.x/2;

    for(unsigned int s = biggerSize/2;s>=0;s>>=1){
        //int index = 2*s*tindex;
        if((tindex<(s))&&((tindex+s)<blockDim.x)){
            sdata[tindex]+=sdata[tindex+s];
        }
        __syncthreads();
        if(s == 0) break;
    }
    if(tindex==0) g_data[size*blockIdx.x] = sdata[0];
}
__global__ void matrixReduction(double *g_data,double* o_data,int size,int biggerSize)
{
    extern __shared__ double sdata[];
    unsigned int tindex = (threadIdx.x);
    unsigned int i = blockIdx.x*size+threadIdx.x;
    sdata[tindex] = g_data[i];
    __syncthreads();

    for(unsigned int s = biggerSize/2;s>=0;s>>=1){
        //int index = 2*s*tindex;
        if((tindex<(s))&&((tindex+s)<blockDim.x)){
            sdata[tindex]+=sdata[tindex+s];
        }
        __syncthreads();
        if(s == 0) break;
    }
    if(tindex==0) o_data[size*blockIdx.x] = sdata[0];

}

__global__ void matrixReductionToVector(double *g_data,double* o_data,int size,int biggerSize)
{
    extern __shared__ double sdata[];
    unsigned int tindex = (threadIdx.x);
    unsigned int i = blockIdx.x*size+threadIdx.x;
    sdata[tindex] = g_data[i];
    __syncthreads();

    for(unsigned int s = biggerSize/2;s>=0;s>>=1){
        //int index = 2*s*tindex;
        if((tindex<(s))&&((tindex+s)<blockDim.x)){
            sdata[tindex]+=sdata[tindex+s];
        }
        __syncthreads();
        if(s == 0) break;
    }
    if(tindex==0) o_data[blockIdx.x] = sdata[0];

}

#endif
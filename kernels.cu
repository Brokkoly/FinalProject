


#include <math.h>



//LSTM Kernels


//Initial: ft = sigmoid(W_f*concatenate(h_t-1,x_t]+bf))
__device__ float sigmoid(float x){
    float retVal = 0;
    retval = 1+exp(x);
    retval = 1/retval;
    return retval;
}

__global__ void firstStep(Matrix X, Matrix H,Matrix Inter,Matrix Wf,Matrix Wi,Matrix Wc,Matrix ft,Matrix it,Matrix Ct1,Matrix ot){

    int threadidx = threadIDx.x;// N threads, N values in output
    int blockId = blockIDx.x;
    int t = blockIDx.y;
    int function = blockIDx.y;
    int Tindex = t*blockDim.x*blockDim.y;
    int Iindex = blockId*blockDim.x*blockDim.y;
    int Windex = threadidx + blockIDx.x*blockDim.x+block
    if(function==0){
        //f_t
        for(int i=0; i < N;i++){

            Inter[Iindex]=Wf[i]
        }


    }
    //Things to calculate:
    //f_t
    //i_t
    //C_t
    //o_t



}
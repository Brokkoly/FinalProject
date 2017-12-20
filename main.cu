#ifdef _WIN32
#  define NOMINMAX 
#endif

#include "kernels.cu"

int main(int argc,char* argv){

    //Initialize weight matrices

    //get inputs from training file
    //get inputs from test file







}

void trainingInstance(float* dx,float* dh, float* dy,float* dyCorrect,float* ddels,float* dgammas,float* dinter,float** dWeights,float** ddeltas,int numX,int numH,int numY,float alpha,float lrate,int dinterSize){

    //firstLayer
    forwardPropagation<<<numH,numX>>>(dx,dinter,dWeights[0],dinterSize);
    matrixReduction<<<numH,numX,numX*sizeof(float)>>>(dinter,dh);
    sigmoidKernel<<<numH>>>(dh);

    //first layer done

    //second layer:
    forwardPropagation<<<numY,numH>>>(dh,dinter,dWeights[1],dinterSize);
    matrixReduction<<<numY,numH,numH*sizeof(float)>>>(dinter,dy);
    sigmoidKernel<<<numY>>>(dy);

    //second layer done

    //backpropagation:
    

    backPropagationFirstKernel<<<numY,numH>>>(dh,dy,dyCorrect,dWeights[1],ddeltas,ddels,alpha,lrate);
    //dim3 grid(numY,numH);
    backPropagationSecondKernelPart1<<<numY,numH>>>(dh,dgammas,dWeights[1],ddels,alpha,lrate);
    matrixReduction<<<numH,numY,numY*sizeof(float)>>>(dgammas,dgammas);
    backPropagationSecondKernelPart2<<<numH,numX>>>(dx,dgammas,dWeights[0],ddeltas,alpha,lrate)






}
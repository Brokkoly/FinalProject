#ifdef _WIN32
#  define NOMINMAX 
#endif

#include "kernels.cu"

int main(int argc,char** argv){

    //Initialize weight matrices

    //get inputs from training file
    //get inputs from test file

    //todo: add main
    float* a = (float*) malloc(2*13*sizeof(float));
    float* b = (float*) malloc(2);
    for(int i = 0; i < 13;i++){
        a[i] = i;
        b[0] +=i;
        a[i+13] = i;
        b[1]+=i;
    }
    a[13] +=100;
    b[1]+=100;
    float* da;
    cudaMalloc((void**)&da,sizeof(float)*26);
    cudaMemcpy(da,a,sizeof(float)*26,cudaMemcpyHostToDevice);
    matrixReduction<<<2,13>>>(da,da);
    cudaMemcpy(a,da,sizeof(float)*26,cudaMemcpyDeviceToHost);

    printf("Device Results: %f,%f\nHost Results: %f,%f\n",a[0],a[13],b[0],b[1]);
    cudaFree(da);
    free(a);
    free(b);



}

void trainingInstance(float* dx,float* dh, float* dy,float* dyCorrect,float* ddels,float* dgammas,float* dinter,float** dWeights,float** ddeltas,int numX,int numH,int numY,float alpha,float lrate,int dinterSize){

    //firstLayer
    forwardPropagation<<<numH,numX>>>(dx,dinter,dWeights[0],dinterSize);
    matrixReduction<<<numH,numX,numX*sizeof(float)>>>(dinter,dh);
    sigmoidKernel<<<1,numH>>>(dh);

    //first layer done

    //second layer:
    forwardPropagation<<<numY,numH>>>(dh,dinter,dWeights[1],dinterSize);
    matrixReduction<<<numY,numH,numH*sizeof(float)>>>(dinter,dy);
    sigmoidKernel<<<1,numY>>>(dy);

    //second layer done

    //backpropagation:
    

    backPropagationFirstKernel<<<numY,numH>>>(dh,dy,dyCorrect,dWeights[1],ddeltas[1],ddels,alpha,lrate);
    //dim3 grid(numY,numH);
    backPropagationSecondKernelPart1<<<numY,numH>>>(dh,dgammas,dWeights[1],ddels,alpha,lrate);
    matrixReduction<<<numH,numY,numY*sizeof(float)>>>(dgammas,dgammas);
    backPropagationSecondKernelPart2<<<numH,numX>>>(dx,dgammas,dWeights[0],ddeltas[0],alpha,lrate);






}
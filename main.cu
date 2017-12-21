#ifdef _WIN32
#  define NOMINMAX 
#endif

#include "kernels.cu"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int hibit(unsigned int n) {
    n |= (n >>  1);
    n |= (n >>  2);
    n |= (n >>  4);
    n |= (n >>  8);
    n |= (n >> 16);
    return n - (n >> 1);
}


unsigned char* read_arrLabels(char* filename, int &len) {
    
    FILE *fp = fopen(filename, "r");
    int temp;
    fscanf(fp,"%d",&temp);
    if(temp<len) len=temp;
    unsigned char *x = (unsigned char*) malloc(len * sizeof(char));
    for (int i = 0; i < len; i++) {
        fscanf(fp, "%f", &x[i]);
    }
    fclose(fp);
    return x;
}
unsigned char* read_arrImage(char* filename, int &len,int &rows,int &cols) {
    FILE *fp = fopen(filename, "r");
    int temp;
    fscanf(fp,"%d",&temp);
    if(temp<len) len=temp;
    fscanf(fp,"%d",&rows);
    fscanf(fp,"%d",&cols);
    unsigned char *x = (unsigned char*) malloc(len*(rows)*(cols) * sizeof(char));
    for (int i = 0; i < len*(rows)*(cols); i++) {
        fscanf(fp, "%f", &x[i]);
    }
    fclose(fp);
    return x;
}


int main(int argc,char** argv){


    int debugLine = 0;
    unsigned char* trainImage;
    unsigned char* trainLabels;
    int len = 1;
    int rows;
    int cols;
    printf("Got to debug # %d\n",++debugLine);
    trainImage = read_arrImage("imagesTrain.txt",len,rows,cols);
    printf("Len: %d\nRows: %d\nCols: %d\n",len,rows,cols);
    printf("Got to debug # %d\n",++debugLine);
    for(int i = 0; i < rows;i++){
        for(int j = 0; j < cols;j++){
            printf("%d ",trainImage[i*cols+j]);
        }
        printf("\n")
    }
    len = 10;
    printf("Got to debug # %d\n",++debugLine);
    trainLabels = read_arrLabels("labelsTrain.txt",len);
    printf("Len: %d\n",len);
    for(int i = 0; i < 10;i++){
        printf("trainLabels[%d]: %d\n",i,trainLabels[i]);
    }
    free(trainLabels);
    free(trainImage);

    //Initialize weight matrices

    //get inputs from training file
    //get inputs from test file

    //todo: add main
    /*float* a = (float*) malloc(2*13*sizeof(float));
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
    printf("hibit: %x\n",hibit(13));
    cudaMalloc(&da,sizeof(float)*26);
    cudaMemcpy(da,a,sizeof(float)*26,cudaMemcpyHostToDevice);
    matrixReductionDestructive<<<2,13,13*sizeof(float)>>>(da,13,hibit(13)<<1);
    cudaMemcpy(a,da,sizeof(float)*26,cudaMemcpyDeviceToHost);

    printf("Device Results: %f,%f\n",a[0],a[13]);
    printf("Host Results: %f,%f\n",b[0],b[1]);
    cudaFree(da);
    free(a);
    free(b);
    */
}


void trainingInstance(float* dx,float* dh, float* dy,float* dyCorrect,float* ddels,float* dgammas,float* dinter,float** dWeights,float** ddeltas,int numX,int numH,int numY,float alpha,float lrate,int dinterSize){

    //firstLayer
    forwardPropagation<<<numH,numX>>>(dx,dinter,dWeights[0],dinterSize);
    matrixReduction<<<numH,numX,numX*sizeof(float)>>>(dinter,dh,1024,hibit(1024));
    sigmoidKernel<<<1,numH>>>(dh);

    //first layer done

    //second layer:
    forwardPropagation<<<numY,numH>>>(dh,dinter,dWeights[1],dinterSize);
    matrixReduction<<<numY,numH,numH*sizeof(float)>>>(dinter,dy,1024,hibit(1024));
    sigmoidKernel<<<1,numY>>>(dy);

    //second layer done

    //backpropagation:
    

    backPropagationFirstKernel<<<numY,numH>>>(dh,dy,dyCorrect,dWeights[1],ddeltas[1],ddels,alpha,lrate);
    //dim3 grid(numY,numH);
    backPropagationSecondKernelPart1<<<numY,numH>>>(dh,dgammas,dWeights[1],ddels,alpha,lrate);
    matrixReduction<<<numH,numY,numY*sizeof(float)>>>(dgammas,dgammas,numY,hibit(numY));
    backPropagationSecondKernelPart2<<<numH,numX>>>(dx,dgammas,dWeights[0],ddeltas[0],alpha,lrate);






}

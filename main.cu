#ifdef _WIN32
#  define NOMINMAX 
#endif

#include "kernels.cu"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <math>
#define NUMY 10
using std::ifstream;
using std::string;
using std::ofstream;

int hibit(unsigned int n) {
    n |= (n >>  1);
    n |= (n >>  2);
    n |= (n >>  4);
    n |= (n >>  8);
    n |= (n >> 16);
    return n - (n >> 1);
}

float* generateDeviceArray(int size){
    float* deviceArr;
    cudaMalloc(deviceArr,size*sizeof(float));
}

float* generateRandomWeights(int size){
    float* weightArr = (float*) malloc(size*sizeof(float));
    for(int i = 0; i < size;i++){
        weightArr[i] = .1
    }
}


unsigned char* read_arrLabels(char* filename, int &len) {
    
    ifstream infile(filename);
    string line;
    int temp;
    getline(infile,line);
    temp = stoi(line);
    if(temp<len) len=temp;
    unsigned char *x = (unsigned char*) malloc(len * sizeof(char));
    for (int i = 0; i < len; i++) {
        //fscanf(fp, "%f", &x[i]);
        getline(infile,line);
        x[i] = stoi(line);
        printf("i = %d,x[i] = %d",i,x[i]);
    }
    infile.close();
    return x;
}
unsigned char* read_arrImage(char* filename, int &len,int &rows,int &cols) {
    //FILE *fp = fopen(filename, "r");
    ifstream infile(filename);
    int temp;
    string line;
    getline(infile,line);
    temp = stoi(line);
    //fscanf(fp,"%d",&temp);
    if(temp<len) len=temp;
    getline(infile,line);
    rows = stoi(line);
    getline(infile,line);
    cols = stoi(line);
    //fscanf(fp,"%d",&rows);
    //fscanf(fp,"%d",&cols);
    unsigned char *x = (unsigned char*) malloc(len*(rows)*(cols) * sizeof(char));
    for (int i = 0; i < len*(rows)*(cols); i++) {
        //fscanf(fp, "%f", &x[i]);
        getline(infile,line);
        x[i] = stoi(line);
    }
    infile.close();
    return x;
}

unsigned char* numToArr(char num){
    unsigned char* x = (unsigned char*) malloc(10*sizeof(unsigned char));
    for(int i = 0; i < 10;i++){
        if(i==num)x[i]=1;
        else x[i]=0;
    }
}

int main(int argc,char** argv){


    int debugLine = 0;
    unsigned char* trainImage;
    unsigned char* trainLabels;
    int len = 1;
    int rows;
    int cols;





    
    //printf("Got to debug # %d\n",++debugLine);
    
    trainImage = read_arrImage("imagesTrain.txt",len,rows,cols);
    printf("Len: %d\nRows: %d\nCols: %d\n",len,rows,cols);
    // for(int i = 0; i < rows;i++){
    //     for(int j = 0; j < cols;j++){
    //         printf("%d ",trainImage[i*cols+j]);
    //     }
    //     printf("\n");
    // }
    len = 1;



    trainLabels = read_arrLabels("labelsTrain.txt",len);
    printf("Len: %d\n",len);
    // for(int i = 0; i < 10;i++){
    //     printf("trainLabels[%d]: %d\n",i,trainLabels[i]);
    // }


    int numX = rows*cols;
    int numY = NUMY;
    int numH = 500;
    float* dx = generateDeviceArray(rows*cols);
    cudaMemcpy(dx,trainImage,rows*cols*sizeof(float),cudaMemcpyHostToDevice);
    float* dh = generateDeviceArray(numH);
    float* dy = generateDeviceArray(NUMY);
    float* dyCorrect = generateDeviceArray(NUMY);
    float* hyCorrect = numToArr(trainLabels[0]);
    cudaMemcpy(dyCorrect,hyCorrect,NUMY*sizeof(float),cudaMemcpyHostToDevice);
    float* ddels = generateDeviceArray(NUMY);
    float* dgammas = generateDeviceArray(numH*NUMY);
    float* dinter = generateDeviceArray(1024*1024);
    float* hWeights1 = generateRandomWeights(numX*numH)
    float* dWeights1 = generateDeviceArray(numX*numH);
    cudaMemcpy(dWeights1,hWeights1,numX*numH*sizeof(float),cudaMemcpyHostToDevice);
    float* hWeights2 = generateRandomWeights(numH*NUMY);
    float* dWeights2 = generateDeviceArray(numH*NUMY);
    cudaMemcpy(dWeights2,hWeights2,numH*NUMY*sizeof(float),cudaMemcpyHostToDevice);
    float* ddeltas1 = generateDeviceArray(rows*cols*numH);
    float* ddeltas2 = generateDeviceArray(numH*NUMY);
    float alpha = .1;
    float lrate = .1;
    int dinterSize = 1024;
    int numX = rows*cols;
    float offset = 1;

    trainingInstance(dx,dh,dy,dyCorrect,ddels,dgammas,dinter,dWeights1,dWeights2,ddeltas1,ddeltas2,numX,numH,numY,offset,alpha,lrate,dinterSize);
    





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


void trainingInstance(float* dx,float* dh, float* dy,float* dyCorrect,float* ddels,float* dgammas,float* dinter,float* dWeights1,float* dWeights2,float* ddeltas1,float* ddeltas2,int numX,int numH,int numY,float offset,float alpha,float lrate,int dinterSize){

    //firstLayer
    forwardPropagation<<<numH,numX>>>(dx,dinter,dWeights1,dinterSize,offset);
    printf("First forward propagation done\n");
    matrixReduction<<<numH,numX,numX*sizeof(float)>>>(dinter,dh,1024,hibit(1024));
    printf("First reduction done\n");
    sigmoidKernel<<<1,numH>>>(dh);
    printf("First sigmoid done\n");
    //first layer done

    //second layer:
    forwardPropagation<<<numY,numH>>>(dh,dinter,dWeights2,dinterSize,offset);
    printf("second forward propagation done\n");
    matrixReduction<<<numY,numH,numH*sizeof(float)>>>(dinter,dy,1024,hibit(1024));
    printf("second reduction done\n");

    sigmoidKernel<<<1,numY>>>(dy);
    printf("second sigmoid done\n");
    //second layer done

    //backpropagation:
    

    backPropagationFirstKernel<<<numY,numH>>>(dh,dy,dyCorrect,dWeights2,ddeltas2,ddels,alpha,lrate);
    //dim3 grid(numY,numH);
    backPropagationSecondKernelPart1<<<numY,numH>>>(dh,dgammas,dWeights1,ddels,alpha,lrate);
    matrixReduction<<<numH,numY,numY*sizeof(float)>>>(dgammas,dgammas,numY,hibit(numY));
    backPropagationSecondKernelPart2<<<numH,numX>>>(dx,dgammas,dWeights1,ddeltas1,alpha,lrate);






}

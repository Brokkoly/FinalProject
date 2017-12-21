#ifdef _WIN32
#  define NOMINMAX 
#endif

#include "kernels.cu"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <random>
//#include <math>
#define NUMY 10
#include <time.h>
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

double* generateDeviceArray(int size){
    double* deviceArr;
    cudaMalloc(&deviceArr,size*sizeof(double));
    return deviceArr;
}

double* generateRandomWeights(int size){
    double* weightArr = (double*) malloc(size*sizeof(double));
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-.1,.1);
    for(int i = 0; i < size;i++){
        weightArr[i] = distribution(generator);
    }
    return weightArr;
}

void printArr(double* arr,int rows,int cols,std::string s){
    std::cout<<s<<std::endl;
    //printf("\n");
    for(int i = 0; i < rows;i++){
        for(int j = 0; j < cols;j++){
            printf(" %lf ", arr[i*cols+j] );
        }
        printf("\n");
    }
}

void printArrFromDevice(double* darr,int rows,int cols,std::string s){
    double* harr = (double*) malloc(rows*cols*sizeof(double));
    cudaMemcpy(harr,darr,rows*cols*sizeof(double),cudaMemcpyDeviceToHost);
    printArr(harr,rows,cols,s);
    free(harr);
}

double* read_arrLabels(char* filename, int &len) {
    
    ifstream infile(filename);
    string line;
    int temp;
    getline(infile,line);
    temp = stoi(line);
    if(temp<len) len=temp;
    int tempint = 0;
    double *x = (double*) malloc(10*len * sizeof(double));
    for (int i = 0; i < len; i++) {
        //fscanf(fp, "%f", &x[i]);
        getline(infile,line);
        tempint = stoi(line);
        //printf("tempint: %d\n",tempint);
        for(int j = 0; j < 10;j++){
            if(tempint == j) {
                x[i*10+j] = 1.0;
            }
            else
            {
                x[i*10+j]=0.0;
            }
            //printf("x[%d*10+%d] = %lf",i,j,x[i*10+j]);
        }
        //printf("i = %d,x[i] = %d",i,x[i]);
    }
    infile.close();
    return x;
}

double* read_arrLabelsTest(char* filename, int &len,int* correct) {
    
    ifstream infile(filename);
    string line;
    int temp;
    getline(infile,line);
    temp = stoi(line);
    if(temp<len) len=temp;
    int tempint = 0;
    double *x = (double*) malloc(10*len * sizeof(double));
    correct = (int*)malloc(len*sizeof(int));
    for (int i = 0; i < len; i++) {
        //fscanf(fp, "%f", &x[i]);
        getline(infile,line);
        tempint = stoi(line);
        correct[i] = tempint;
        //printf("tempint: %d\n",correct[i]);
        for(int j = 0; j < 10;j++){
            if(tempint == j) {
                x[i*10+j] = 1.0;
            }
            else
            {
                x[i*10+j]=0.0;
            }
            //printf("x[%d*10+%d] = %lf",i,j,x[i*10+j]);
        }
        //printf("i = %d,x[i] = %d",i,x[i]);
    }
    infile.close();
    return x;
}
char* read_arrImage(char* filename, int &len,int &rows,int &cols) {
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
    char *x = (char*) malloc(len*(rows)*(cols) * sizeof(char));
    for (int i = 0; i < len*(rows)*(cols); i++) {
        //fscanf(fp, "%f", &x[i]);
        getline(infile,line);
        x[i] = stoi(line);
    }
    infile.close();
    return x;
}

/*
double* numToArr(char num){
    double* x = (double*) malloc(10*sizeof(double));
    for(int i = 0; i < 10;i++){
        if(i==num)x[i]=1;
        else x[i]=0;
    }

}
*/

void trainingInstance(double* dx,double* dh, double* dy,double* dyCorrect,double* ddels,double* dgammas,double* dinter,double* dWeights1,double* dWeights2,double* ddeltas1,double* ddeltas2,int numX,int numH,int numY,double offset,double alpha,double lrate,int dinterSize){
    //double* testOutput = (double*)malloc(10*sizeof(double));
    //firstLayer
    // printArrFromDevice(dx,1,numX);
    forwardPropagation<<<numH,numX>>>(dx,dinter,dWeights1,dinterSize,0);
    // printf("dinter\n");
    // printArrFromDevice(dinter,2,1024);
    // printArrFromDevice(dWeights1,numH,numX);
    // printf("First forward propagation done\n");
    matrixReductionToVector<<<numH,numX,numX*sizeof(double)>>>(dinter,dh,1024,hibit(1024));
    // printArrFromDevice(dh,1,numH);
    // printf("First reduction done\n");

    sigmoidKernel<<<1,numH>>>(dh);
    // printf("First sigmoid done\n");
    //first layer done
    // printArrFromDevice(dh,1,numH);
    //second layer:
    forwardPropagation<<<numY,numH>>>(dh,dinter,dWeights2,dinterSize,offset);
    // printf("second forward propagation done\n");
    // printArrFromDevice(dWeights2,numY,numH);
    matrixReductionToVector<<<numY,numH,numH*sizeof(double)>>>(dinter,dy,1024,hibit(1024));
    // printf("second reduction done\n");
    // printArrFromDevice(dy,1,numY);
    sigmoidKernel<<<1,numY>>>(dy);
    // printArrFromDevice(dy,1,numY);
    // printf("second sigmoid done\n");
    //second layer done

    //backpropagation:
    
    // printf("dyCorrect then dy\n");
    //printArrFromDevice(dy,1,numY,"OutputY: ");
    //printArrFromDevice(dyCorrect,1,numY,"CorrectY ");
    //printArrFromDevice(dy,1,numY);
    backPropagationFirstKernel<<<numY,numH>>>(dh,dy,dyCorrect,dWeights2,ddeltas2,ddels,alpha,lrate);
    //printf("Deltas for W2: \n");
    //printArrFromDevice(ddeltas2,numY,numH);
    //dim3 grid(numY,numH);
    //printf("Dels: \n");
    //printArrFromDevice(ddels,1,numY);
    backPropagationSecondKernelPart1<<<numY,numH>>>(dh,dgammas,dWeights1,ddels,alpha,lrate);
    //printf("dgammas\n");

    //printArrFromDevice(dgammas,numH,numY);

    matrixReduction<<<numH,numY,numY*sizeof(double)>>>(dgammas,dgammas,numY,hibit(numY));
    backPropagationSecondKernelPart2<<<numH,numX>>>(dx,dgammas,dWeights1,ddeltas1,alpha,lrate);
    //printf("Deltas for W1: \n");
    //printArrFromDevice(ddeltas1,numH,numX,"W1 Deltas");
    //free(testOutput);
}


void longTraining(int len,double* trainLabels,char* trainImage,int epochs,double* dx,double* dh, double* dy,double* dyCorrect,double* ddels,double* dgammas,double* dinter,double* dWeights1,double* dWeights2,double* ddeltas1,double* ddeltas2,int numX,int numH,int numY,double offset,double alpha,double lrate,int dinterSize){


    double* trainImageDouble = (double*) malloc(numX*sizeof(double));
    double* trainLabelsInner = (double*) malloc(numY*sizeof(double));
    for(int q = 0; q < epochs;q++){
        for(int i = 0; i < len;i++){
            for(int j = 0; j < numX;j++){
                trainImageDouble[j] = (double)trainImage[j+i*numX];
            }
            for(int j = 0; j < numY;j++){
                trainLabelsInner[j] = trainLabels[j+i*numY];
                //printf(" %lf ",trainLabelsInner[j]);

            }
            //printf("\n");
            cudaMemcpy(dx,trainImageDouble,numX*sizeof(double),cudaMemcpyHostToDevice);
            //free(hyCorrect);
            //hyCorrect = numToArr(trainLabels[i]);
            cudaMemcpy(dyCorrect,trainLabelsInner,numY*sizeof(double),cudaMemcpyHostToDevice);
            trainingInstance(dx,dh,dy,dyCorrect,ddels,dgammas,dinter,dWeights1,dWeights2,ddeltas1,ddeltas2,numX,numH,numY,offset,alpha,lrate,dinterSize);
        }
    }
    //free(hyCorrect);
    free(trainImageDouble);
    free(trainLabelsInner);
}

void testingInstance(double* dx,double* dh, double* dy,double* dinter,double* dWeights1,double* dWeights2,int numX,int numH,int numY,double offset,int dinterSize){
    forwardPropagation<<<numH,numX>>>(dx,dinter,dWeights1,dinterSize,0);
    matrixReductionToVector<<<numH,numX,numX*sizeof(double)>>>(dinter,dh,1024,hibit(1024));
    sigmoidKernel<<<1,numH>>>(dh);
    forwardPropagation<<<numY,numH>>>(dh,dinter,dWeights2,dinterSize,offset);
    matrixReductionToVector<<<numY,numH,numH*sizeof(double)>>>(dinter,dy,1024,hibit(1024));
    sigmoidKernel<<<1,numY>>>(dy);
}

void testing(int len,double* testLabels,char* testImage,double* results,double* dx,double* dh, double* dy,double* dinter,double* dWeights1,double* dWeights2,int numX,int numH,int numY,double offset,int dinterSize){

    double* testImageDouble = (double*) malloc(numX*sizeof(double));
    double* testLabelsInner = (double*) malloc(numY*sizeof(double));
    for(int i = 0; i < len;i++){
        for(int j = 0; j < numX;j++){
                testImageDouble[j] = (double)testImage[j+i*numX];
            }
        for(int j = 0; j < numY;j++){
                testLabelsInner[j] = testLabels[j+i*numY];
                //printf(" %lf ",trainLabelsInner[j]);
            }
            //printf("\n");
        cudaMemcpy(dx,testImageDouble,numX*sizeof(double),cudaMemcpyHostToDevice);
            //free(hyCorrect);
            //hyCorrect = numToArr(trainLabels[i]);
        //cudaMemcpy(dyCorrect,testLabelsInner,numY*sizeof(double),cudaMemcpyHostToDevice);
        testingInstance(dx,dh,dy,dinter,dWeights1,dWeights2,numX,numH,numY,offset,dinterSize);

        cudaMemcpy(testLabelsInner,dy,numY*sizeof(double),cudaMemcpyDeviceToHost);
        for(int j = 0;j < numY;j++){
            results[j+i*numY] = testLabelsInner[j];
        }
    }
    free(testImageDouble);
    free(testLabelsInner);
}




int main(int argc,char** argv){


    char* trainImage;
    double* trainLabels;
    int len = 10000;
    int rows;
    int cols;
	srand(time(NULL));
    char* testImage;
    double* testLabels;
    int* correct;
    int epochs = std::stoi(argv[1]);

    
    //printf("Got to debug # %d\n",++debugLine);
    //trainImage = (char* )malloc(10*sizeof(char));
    trainImage = read_arrImage("imagesTrain.txt",len,rows,cols);
    printf("Len: %d\nRows: %d\nCols: %d\n",len,rows,cols);
    /*
    for(int i = 0; i < rows;i++){
        for(int j = 0; j < cols;j++){
            printf("%d ",trainImage[i*cols+j]);
        }
        printf("\n");
    }
    */
    len = 10000;



    trainLabels = read_arrLabels("labelsTrain.txt",len);
    //trainLabels = (double*) malloc(2*sizeof(double));
    printf("Len: %d\n",len);


    /*
    for(int i = 0; i < len;i++){
        printf("trainLabels[%d]: \n",i);
        for(int j = 0; j < NUMY;j++){
            printf(" %lf ",trainLabels[i*NUMY+j]);
        }
        printf("\n");
        
    }
    */
    int testLen = 10000;
    testImage = read_arrImage("imagesTest.txt",testLen,rows,cols);
    testLabels = read_arrLabels("labelsTest.txt",testLen);
    correct = (int*) malloc(testLen*sizeof(int));

	for(int i = 0; i < testLen;i++){
		for(int j =0 ; j < NUMY;j++){
		if(testLabels[j+i*NUMY]>.9){
		correct[i] = j;
			break;
			}	
		}
	}
    //int numX = 10;
    printf("TestLen: %d\n",testLen);
	printf("Testing For correct: %d\n",correct[0]);

    int numX = rows*cols;
    int numY = NUMY;
    int numH = 50;


    
    double* dx = generateDeviceArray(numX);
    cudaMemcpy(dx,trainImage,numX*sizeof(double),cudaMemcpyHostToDevice);
    //cudaMemcpy(dx,trainImageDouble,numX*sizeof(double),cudaMemcpyHostToDevice);
    //free(trainImageDouble);
    double* dh = generateDeviceArray(numH);
    double* dy = generateDeviceArray(NUMY);
    double* dyCorrect = generateDeviceArray(NUMY);
    double* hyCorrect = (double*)malloc(numY*sizeof(double));//numToArr(trainLabels[0]);
    //cudaMemcpy(dyCorrect,hyCorrect,NUMY*sizeof(double),cudaMemcpyHostToDevice);
    double* ddels = generateDeviceArray(NUMY);
    double* dgammas = generateDeviceArray(numH*NUMY);
    double* dinter = generateDeviceArray(1024*1024);
    double* hWeights1 = generateRandomWeights(numX*numH);
    //printArr(hWeights1,numH,numX,"");
    double* dWeights1 = generateDeviceArray(numX*numH);
    cudaMemcpy(dWeights1,hWeights1,numX*numH*sizeof(double),cudaMemcpyHostToDevice);
    double* hWeights2 = generateRandomWeights(numH*NUMY);
    double* dWeights2 = generateDeviceArray(numH*NUMY);
    cudaMemcpy(dWeights2,hWeights2,numH*NUMY*sizeof(double),cudaMemcpyHostToDevice);
    double* ddeltas1 = generateDeviceArray(numX*numH);
    double* ddeltas2 = generateDeviceArray(numH*NUMY);
    double alpha = .1;
    double lrate = .01;
    int dinterSize = 1024;
    double offset = .1;

    double* results =(double*) malloc(testLen*NUMY*sizeof(double));
    int* bestMatch = (int*)malloc(testLen*sizeof(int));


    longTraining(len,trainLabels,trainImage,epochs,dx,dh,dy,dyCorrect,ddels,dgammas,dinter,dWeights1,dWeights2,ddeltas1,ddeltas2,numX,numH,numY,offset,alpha,lrate,dinterSize);

    testing(testLen,testLabels,testImage,results,dx,dh,dy,dinter,dWeights1,dWeights2,numX,numH,numY,offset,dinterSize);
	
    int* dbestMatch;
	cudaMalloc(&dbestMatch,testLen*sizeof(int));
    //double* dresults= 
	double* dresults =     generateDeviceArray(testLen*NUMY);

    int numThreads = 1024;
    int numBlocks = testLen/1024 + 1;
cudaMemcpy(dresults,results,sizeof(double)*testLen*NUMY,cudaMemcpyHostToDevice);
    bestChoiceKernel<<<numBlocks,numThreads>>>(dresults,dbestMatch,testLen,NUMY);
   cudaMemcpy(bestMatch,dbestMatch,testLen*sizeof(int),cudaMemcpyDeviceToHost);
    int err = 0;
    int right = 0;
  // int temp = bestMatch[0];
//double temp2 = results[0];
//int temp = correct[0];
//printf("%d\n",temp);
//printf("%lf\n",temp2);
//printf("%d\n",correct[0]);
int tempMax = 0;
int maxIndex = 0;
for(int i =0; i < 10;i++){
    tempMax =0;
    maxIndex = 0;
    for(int j = 0; j < NUMY;j++){
        if(results[j+numY*i]>tempMax){
            tempMax = results[j+numY*i];
            maxIndex = j;
        }
	   printf(" %f ",results[j+NUMY*i]);
    }
    printf("MAXINDEX: %d\n",maxIndex);
    printf("\n");
}

    for(int i = 0; i < testLen;i++){
        if(i<10){
            printf("best: %d\ncorrect: %d\n",bestMatch[i],correct[i]);
	
        }
        if(bestMatch[i]!=correct[i]){
            err++;
        }
        else{
            right++;
        }
    }
  
    printf("# correct: %d\n",right);
    printf("# wrong: %d\n",err);
    //trainingInstance(dx,dh,dy,dyCorrect,ddels,dgammas,dinter,dWeights1,dWeights2,ddeltas1,ddeltas2,numX,numH,numY,offset,alpha,lrate,dinterSize);




    free(hyCorrect);
    free(hWeights2);
    free(hWeights1);
    cudaFree(dx);
    cudaFree(dh);
    cudaFree(dy);
    cudaFree(dyCorrect);
    cudaFree(ddels);
    cudaFree(dgammas);
    cudaFree(dinter);
    cudaFree(ddeltas1);
    cudaFree(ddeltas2);
    cudaFree(dWeights2);
    cudaFree(dWeights1);
cudaFree(dresults);
cudaFree(dbestMatch);
    free(testLabels);
    free(testImage);
	free(correct);
free(results);
free(bestMatch);
    free(trainLabels);
    free(trainImage);
    
    //Initialize weight matrices

    //get inputs from training file
    //get inputs from test file

    //todo: add main
    /*double* a = (double*) malloc(2*13*sizeof(double));
    double* b = (double*) malloc(2);
    for(int i = 0; i < 13;i++){
        a[i] = i;
        b[0] +=i;
        a[i+13] = i;
        b[1]+=i;
    }
    a[13] +=100;
    b[1]+=100;
    double* da;
    printf("hibit: %x\n",hibit(13));
    cudaMalloc(&da,sizeof(double)*26);
    cudaMemcpy(da,a,sizeof(double)*26,cudaMemcpyHostToDevice);
    matrixReductionDestructive<<<2,13,13*sizeof(double)>>>(da,13,hibit(13)<<1);
    cudaMemcpy(a,da,sizeof(double)*26,cudaMemcpyDeviceToHost);

    printf("Device Results: %f,%f\n",a[0],a[13]);
    printf("Host Results: %f,%f\n",b[0],b[1]);
    cudaFree(da);
    free(a);
    free(b);
    */
}




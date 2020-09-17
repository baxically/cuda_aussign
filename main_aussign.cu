/******************************************************************************
 *cr
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <functional>
#include <iterator> 
#include <iterator>
#include <time.h>
#include "kernel_retrain.cu"


using namespace std;
vector<float> linSpace(float start_in, float end_in, int num_in);
void printLinspace(vector<float> v);


int main ()
{

    time_t load_start, load_stop;
    time_t hv_start, hv_stop; 
    time_t train_start, train_stop;
    time_t test_start, test_stop; 
    time_t total_start, total_stop;
	
	time(&total_start);

	//*****************************************************
    //******************** Model Parameter ****************
	//*****************************************************

	int M=11;
	int D=10000;
	float ephsilon = 0.01;
	int max_epoch=30;
	int min_epoch=10;
	
	//*****************************************************
    //******************** Dataset Parameter ****************
	//*****************************************************
	
	
	int numClasses=95;
    int numTrainSampOrig=1796;
	int numTestSamples=769;
	int numFeatures=22;
	
	int numValidSamples=0;
	int numTrainSamples=numTrainSampOrig-numValidSamples;
	
	
	
	//**********************************************************
	//********* Initialize Host and Device Variables ***********
	//**********************************************************
	


 
	float lMax;
    float lMin;
	vector<float>L;
	float accuracy;
	
	float *trainX_h, *validX_h,*testX_h;
	int *trainY_h, *validY_h, *testY_h, *Classes_h;
	int *LD_h, *ID_h, *ClassHV_h;
	float *L_h;
   
	size_t trainX_sz, trainY_sz, validX_sz, validY_sz, testX_sz, testY_sz, L_sz, trainQ_sz;
	
	float *trainX_d,*validX_d, *testX_d;
	int *trainQ_d,*validY_d, *trainY_d,*testY_d;
	float *L_d;
	int *LD_d, *ID_d, *ClassHV_d; 

	
	
	//*****************************************************
	//*********** Initialize data loading variables********
	//*****************************************************
	ifstream fin;
	ifstream ftestin;
    ofstream fout;
	vector<vector<float> > trainset_array;
    vector<int> trainset_labels(numTrainSampOrig+1);
    vector<vector<float> > testset_array;
    vector<int> testset_labels(numTestSamples+1);
	int row = 0;
	vector<float> rowArray(numFeatures);
	
	
	
	
	
	
	
	//******************************************
	//***** Dynamic Host Memory Allocation******
	//******************************************
	
	trainX_h = (float*) malloc( sizeof(float)*numTrainSamples*numFeatures );
	trainY_h = (int*) malloc( sizeof(int)*numTrainSamples );
	
	validX_h = (float*) malloc( sizeof(float)*numValidSamples*numFeatures );
	validY_h = (int*) malloc( sizeof(int)*numValidSamples );
	
	testX_h = (float*) malloc( sizeof(float)*numTestSamples*numFeatures );
	testY_h = (int*) malloc( sizeof(int)*numTestSamples );
	
	L_h = (float*) malloc( sizeof(float)*M );
	LD_h = (int*) malloc( sizeof(int)*M*D );
	ID_h = (int*) malloc( sizeof(int)*numFeatures*D );
	
	ClassHV_h = (int*) malloc( sizeof(int)*numClasses*D );
	
	
	
	
	//********************************************
	//***** Dynamic Device Memory Allocation******
	//********************************************
	
	trainX_sz= numTrainSamples*numFeatures*sizeof(float);
	trainY_sz=numTrainSamples*sizeof(int);
	validX_sz= numValidSamples*numFeatures*sizeof(float);
	validY_sz=numValidSamples*sizeof(int);
	L_sz=M*sizeof(float);
	trainQ_sz=numTrainSamples*numFeatures*sizeof(int);
	testX_sz= numTestSamples*numFeatures*sizeof(float);
	testY_sz=numTestSamples*sizeof(int);

	
	cudaMalloc((void **)&trainX_d, trainX_sz);
	cudaMalloc((void **)&trainY_d, trainY_sz);
	cudaMalloc((void **)&validX_d, trainX_sz);
	cudaMalloc((void **)&validY_d, trainY_sz);
	cudaMalloc((void **)&testX_d, testX_sz);
	cudaMalloc((void **)&testY_d, testY_sz);
	
	cudaMalloc((void **)&L_d, L_sz);
	cudaMalloc((void **)&LD_d, M*D*sizeof(int));
	cudaMalloc((void **)&ID_d, numFeatures*D*sizeof(int));
	
	cudaMalloc((void **)&trainQ_d, trainQ_sz);
	cudaMalloc((void **)&ClassHV_d, numClasses*D*sizeof(int));
	
	
	
	
	
	
	
	//**************************************
	//************ Data Loading **********
	//************************************
	float v=0.0;
	int vl=0;
	time(&load_start);
	printf("\nSetting up the problem...\n"); fflush(stdout);
	
	fin.open("aussign_train.csv");
    if(!fin.is_open())
    {
        printf( "Error: Can't open file containind training X dataset"  );
    }
    else
    {	printf("\nloading train data..\n");
        while(!fin.eof())
        {
            
            if(row >= numTrainSampOrig)
            {
                break;
            }
            
            fin >> vl;
			trainset_labels[row]=vl;
			//printf("\nSample:%d\tLabel:%d\n\n",row, trainset_labels[row]);
			
            trainset_array.push_back(rowArray);
			
            
            for(int col = 0; col < numFeatures; col++)
            {
                fin.ignore(50000000, ',');
                fin >> v;
				//while (v==122.00)
				//{fin >>v;}
				trainset_array[row][col]=v;
				//if (col<20) printf(" %f",trainset_array[row][col]);
            }
			fin.ignore();
            row++;
        }
		
    }
	fin.close();
	row=0;
	ftestin.open("aussign_test.csv");
    if(!ftestin.is_open())
    {
        printf("Error: Can't open file containind training X dataset");
    }
    else
    {	printf("\nloading test data..\n");
        while(!ftestin.eof())
        {
            
            if(row >= numTestSamples)
            {
                break;
            }
            
            ftestin >> vl;
			testset_labels[row]=vl;
			//printf("\nSample:%d\tLabel:%d\n\n",row, testset_labels[row]);
            testset_array.push_back(rowArray);
			
            
            for(int col = 0; col < numFeatures; col++)
            {
                ftestin.ignore(50000000, ',');
                ftestin >>v;
				testset_array[row][col]=v;
				//printf("\t%f",testset_array[row][col]);
            }
			fin.ignore();
			//fin.ignore();
            row++;
			//printf("checkpoint1: %d",row);
        }
		
    }
	ftestin.close();
	
    for (int i=0; i < numTrainSamples*numFeatures; i++) 
	{ 	
		trainX_h[i] = trainset_array[i/numFeatures][int(i%numFeatures)]; 
	}
	for (int i=0; i < numTrainSamples; i++) { trainY_h[i] = trainset_labels[i]; }
	
	for (int i=0; i < numValidSamples*numFeatures; i++) 
	{ 	
		validX_h[i] = trainset_array[(i/numFeatures)+numTrainSamples][int(i%numFeatures)]; 
	}
	for (int i=0; i < numValidSamples; i++) { validY_h[i] = trainset_labels[(i+numTrainSamples)]; }
	
	
    for (int i=0; i < numTestSamples*numFeatures; i++) 
	{ 	
		testX_h[i] = testset_array[i/numFeatures][int(i%numFeatures)]; 
	}
	for (int i=0; i < numTestSamples; i++) { testY_h[i] = testset_labels[i]; }
	
	
	

	//***********************************************
	//********* Copy dataset to Device***************
	//***********************************************
	
	cudaMemcpy(trainX_d, trainX_h, trainX_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(trainY_d, trainY_h, trainY_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(validX_d, validX_h, validX_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(validY_d, validY_h, validY_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(testX_d, testX_h, testX_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(testY_d, testY_h, testY_sz, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	
	printf("\ndata loading done ..\n");
	
	time(&load_stop);
	float load_seconds = load_stop - load_start;
	printf("Loading time in seconds: %fn", load_seconds);
	
	
	

	
	
	//******************************************
	//******** Level Hypervector ***************
	//******************************************
	
	time(&hv_start);
	printf("\nSetting up Level and Identity Hypervector....\n");
	
	//*******Defining Quantization Levels *****
	
	L_h = (float*) malloc( sizeof(float)*M );
	LD_h = (int*) malloc( sizeof(int)*M*D );
	ID_h = (int*) malloc( sizeof(int)*numFeatures*D );
	
	lMin= *min_element(trainset_array[0].begin(),trainset_array[0].end());
    lMax= *max_element(trainset_array[0].begin(),trainset_array[0].end());
	L = linSpace(lMin, lMax, M);
	for (int i=0; i<M; i++)
	{
		L_h[i]=L[i];
	}

	//*********Setting up Level Hypervector*********
	
	
	
	for (int i=0; i<D; i++) {LD_h[i]=int(rand()%2);}
	int nAlter[D];
	for (int i=1; i<D; i++)
	{
		nAlter[i]=rand()%10000;
	}
	//random_shuffle(nAlter.begin(), nAlter.end());	
	int jAlter;
	
	for (int i=1; i<M; i++)
	{
		for (int j=0; j<D; j++)
		{
			LD_h[i*D+j]=LD_h[(i-1)*D+j];
		}
		for (int j=0; j<ceil(D/M); j++)
		{
			jAlter=nAlter[int((i-1)*ceil(D/M)+j)];
			LD_h[(i*D)+jAlter]=int(LD_h[(i*D)+jAlter]==0);
		}
	}
	
	//*********test to see if LD is being populated properly***********
	
    int LD_test1=0;
    int LD_test2=0;
    for(int jtest = 0; jtest < D; jtest++)
    {
      LD_test1=LD_test1+ (LD_h[0+jtest]^LD_h[D+jtest]) ; //FIX ME: print out all 0s
      LD_test2=LD_test2+ (LD_h[5*D+jtest]^LD_h[0*D+jtest]) ;
    }
    cout <<"LDTEST1: "<< LD_test1<< endl;
    cout << "LDTEST2: "<<LD_test2<< endl;
	
	
	
	
	
	//************************************************
	//******* Creating Identity Hypervector ID *******
	//************************************************
	

	for (int i=0; i<numFeatures; i++)
	{
		for (int j=0; j<D; j++)
		{
			ID_h[i*D+j]=int(rand()%2);
		}	
	}
	

	//******test to see if ID is being populated properly********
	
    int ID_test = 0;
    for(int j = 0; j < D; j++)
    {
        ID_test=ID_test+ (ID_h[D+j]^ID_h[j]) ; 
	}
    cout <<"IDTEST:"<<ID_test<< endl;
	
	cudaMemcpy(L_d, L_h, M*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(LD_d, LD_h, M*D*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(ID_d, ID_h, numFeatures*D*sizeof(int), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	
			
	printf("Creating level and Identity Hypervector Done\n");
	time(&hv_stop);
	float hv_seconds = hv_stop - hv_start;
	printf("Hypervector creation time in seconds: %fn", hv_seconds);
	
	
	


	//*****************************************
	//****************  Training **************
	//*****************************************

	time(&train_start);
	printf("\nTraining...\n");
	//copy trainX,trainY L to device___________

	

	Classes_h=(int*) malloc( sizeof(int)*numClasses );
	for(int i=0; i<numClasses; i++)
	{
		Classes_h[i]=0;
	}
	Training_HV(trainX_d,trainY_d,validX_d,validY_d,L_d,trainQ_d,numTrainSamples,numValidSamples,numFeatures,numClasses, M,D,LD_d,ID_d,ClassHV_d, Classes_h, ephsilon, max_epoch,min_epoch);
	

	cudaMemcpy(ClassHV_h, ClassHV_d, numClasses*D*sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	time(&train_stop);
	float train_seconds = train_stop - train_start;
	printf("Training time in seconds: %fn", train_seconds);
	
	
	//***************Teasting  ClassHyper Vectors ***********
	
	int Class_test1=0;
	int Class_test2=0;
	
	for (int i=0;i<D; i++)
	{
		Class_test1=Class_test1+(ClassHV_h[0*D+i]^ClassHV_h[1*D+i]);
		Class_test2=Class_test2+(ClassHV_h[0*D+i]^ClassHV_h[5*D+i]);	
	}
	printf("\n Classtest1=%d", Class_test1);
	printf("\n Classtest2=%d", Class_test2);

 
	
	
	
	
	//********************************
	//*********   Testing  ***********
	//********************************
	
	
	time(&test_start);
	
	accuracy=TestingAccuracy(testX_d, testY_d, ClassHV_d, L_d,LD_d,ID_d, D,M, numTestSamples, numFeatures,numClasses);
	printf("\naccuracy=%f\n",accuracy);
	
	printf("\n Testing done......\n");
	
	time(&test_stop);
	float test_seconds = test_stop - test_start;
	printf("\nTesting time in seconds: %fn", test_seconds);
	
	
	
	//**********************************************************************
	//************** Free dynamically allocated memory and finish **********
	//**********************************************************************
	
	time(&total_stop);
	float elapsed_time = total_stop - total_start;
	printf("\nTotal elapsed time in seconds: %fn", elapsed_time);	
 
	cudaFree(trainX_d);
	cudaFree(trainY_d);
	cudaFree(trainQ_d);
	cudaFree(testX_d);
	cudaFree(testY_d);
	
	free(trainX_h);
	free(trainY_h);
	free(testX_h);
	free(testY_h);
	free(L_h);
	free(LD_h);
	free(ID_h);
	free (ClassHV_h);

	
	cudaFree(L_d);
	cudaFree(LD_d);
	cudaFree(ID_d);
	cudaFree(ClassHV_d);
	
}

vector<float> linSpace(float start_in, float end_in, int num_in)
{
    vector<float> linspaced;
    
    float start = start_in;
    float end = end_in;
    int num = num_in;
    
    if(num == 0)
    {
        return linspaced;
    }
    
    if(num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }
    
    float delta = (end - start) / (num - 1);
    
    for(int i = 0; i < num - 1; i++)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end);
    return linspaced;
}

void printLinspace(vector<float> v)
{
    cout << "size: " << v.size() << endl;
    for(int i=0; i< v.size(); i++)
    {
        cout << v[i] << " ";
    }
    cout << endl;
}
/******************************************************************************

 *cr
 ******************************************************************************/

#include <stdio.h>
#include <math.h>

   

__global__ void QuantKernel(int n, int M, float *trainX_d,int *trainQ_d, float *L_d) {

	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int indexMin=0;
	float quantMin=abs(trainX_d[tid]-L_d[indexMin]);
	if(tid<n)
	{
		for (int i=1;i<M;i++)
		{
			if ( (abs(trainX_d[tid]-L_d[i]))<quantMin)
			{
				quantMin=abs(trainX_d[tid]-L_d[i]);
				indexMin=i;
			}
		}
		trainQ_d[tid]=indexMin;
	}

}

__global__ void SampleHVKernel(int n_SHV, int D, int numTrainSamples, int numFeatures, int *trainQ_d, int *SampleHV_d, int *LD_d, int *ID_d) 
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int samp_ind= int(tid/D);
	int D_ind=int(tid%D);
	if (tid<n_SHV)
	{
		SampleHV_d[tid]=0;
		
		for (int i_s=0; i_s<numFeatures; i_s++)
		{
			int LD_ind=trainQ_d[samp_ind*numFeatures+i_s];
			SampleHV_d[tid]=SampleHV_d[tid]+(LD_d[LD_ind*D+D_ind]^ID_d[i_s*D+D_ind]);
		}
		SampleHV_d[tid]=int(SampleHV_d[tid]>=(numFeatures/2));

	}
}

__global__ void ClassHVAddKernel(int n_SHV, int D, int *trainY_d, int *SampleHV_d, int *ClassHVdec_d, int *Classes_d) {
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int s_ind=tid/D;
	int c_ind=trainY_d[s_ind];
	int D_ind=tid%D;

	
	if (tid<n_SHV)
	{
		atomicAdd(&ClassHVdec_d[c_ind*D+D_ind],SampleHV_d[tid]);
		
		if (D_ind==0)
		{
			atomicAdd(&Classes_d[c_ind],1);
		}
	}
}


__global__ void ClassHVKernel(int ClassHV_sz, int D, int numClasses,int *ClassHV_d, int *ClassHVdec_d,int *Classes_d) {
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int cid=int(tid/D);
	if (tid<ClassHV_sz)
	{
		ClassHV_d[tid]=int(!(ClassHVdec_d[tid]<(Classes_d[cid]/2)));
	}
}

__global__ void CheckSumKernel(int n_C, int *QueryHV_d,int *ClassHV_d,int *CheckSumHV_d,int numTestSample,int numClasses, int D)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int samp_id=int(tid/numClasses);
	int class_id=int(tid%numClasses);
	CheckSumHV_d[tid]=0;
	if (tid<n_C)
	{
		for (int i=0; i<D; i++)
		{
			CheckSumHV_d[tid]=CheckSumHV_d[tid]+(QueryHV_d[samp_id*D+i]^ClassHV_d[class_id*D+i]);
		}
	}
}

__global__ void TestingKernel(int n_T,int *CheckSumHV,int *testLebel, int *testY, int *AccuVector,int numTestSample,int numClasses)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int samp_id=tid;
	int minIndex=0;
	int minValue = CheckSumHV[samp_id*numClasses];
	
	if (tid<n_T)
	{
		for (int i=1; i<numClasses; i++)
		{
			if (minValue>CheckSumHV[samp_id*numClasses+i])
			{
				minValue=CheckSumHV[samp_id*numClasses+i];
				minIndex=i;
			}
			
		}
		testLebel[tid]=minIndex;
		AccuVector[tid]=int(testLebel[tid]==testY[tid]);

	}
}

__global__ void RetraindKernel(int n_SHV,int D,int *trainY_d,int *RetrainLebel_d, int *SampleHV_d,int *ClassHVdec_d,int *Classes_d)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int s_ind=tid/D;
	int c_ind=trainY_d[s_ind];
	int RTc_ind=RetrainLebel_d[s_ind];
	int D_ind=tid%D;

	
	if (tid<n_SHV)
	{
		if (c_ind!=RTc_ind)
		{
			atomicAdd(&ClassHVdec_d[c_ind*D+D_ind],SampleHV_d[tid]);
			atomicAdd(&ClassHVdec_d[RTc_ind*D+D_ind],-SampleHV_d[tid]);
			if (D_ind==0)
			{
				atomicAdd(&Classes_d[c_ind],1);
				atomicAdd(&Classes_d[RTc_ind],-1);
			}
				
		}
	}
}







float TestingAccuracy(float *testX_d, int *testY_d, int* ClassHV_d, float *L_d,int *LD_d, int *ID_d, int D,int M, int numTestSamples, int numFeatures,int numClasses)
{


	//******************************************************
    //**********Initialize thread block dimensions**********
	//******************************************************
	
	
	const unsigned int BLOCK_SIZE = 1024;
	
	int *QueryHV_d, *testQ_d, *AccuVector_d,*AccuVector_h,*testLebel_h, *testLebel_d,*CheckSumHV_d;
	int n=numTestSamples*numFeatures;
	int n_SHV=numTestSamples*D;
	int n_C=numTestSamples*numClasses;
	int n_T=numTestSamples;
	float Accuracy_d=0.0;
	
	//**************************************************
	//********** Dynamic Memory Acclocation ************
	//**************************************************
	
	cudaMalloc((void **)&testQ_d, numTestSamples*D*sizeof(int));
	cudaMalloc((void **)&QueryHV_d, numTestSamples*D*sizeof(int));
	cudaMalloc((void **)&AccuVector_d, numTestSamples*sizeof(int));
	cudaMalloc((void **)&testLebel_d, numTestSamples*sizeof(int));
	cudaMalloc((void **)&CheckSumHV_d, numTestSamples*numClasses*sizeof(int));
	testLebel_h = (int*) malloc( sizeof(int)*numTestSamples );
	AccuVector_h = (int*) malloc( sizeof(int)*numTestSamples );
	
	
	//**********************************************
	//******** Quantizing All Test Sample **********
	//**********************************************
	
	QuantKernel<<<ceil((float)n/BLOCK_SIZE), BLOCK_SIZE>>>(n,M,testX_d,testQ_d,L_d);
	cudaDeviceSynchronize();
	
	
	
	
	
	//*********************************************
	//********Creating Query Hypervector***********
	//*********************************************
	
	SampleHVKernel<<<ceil((float)n_SHV/BLOCK_SIZE), BLOCK_SIZE>>>(n_SHV,D,numTestSamples, numFeatures,testQ_d,QueryHV_d,LD_d, ID_d);
	cudaDeviceSynchronize();

	
	
	
	
	
	//****************************************************
	//************* Testing :Similerity Check ************
	//****************************************************


	CheckSumKernel<<<ceil((float)n_C/BLOCK_SIZE), BLOCK_SIZE>>>(n_C, QueryHV_d,ClassHV_d,CheckSumHV_d,numTestSamples,numClasses, D);
	cudaDeviceSynchronize();
	
	//****************************************************
	//************ Testing: Predict Class ****************
	//****************************************************
	
	TestingKernel<<<ceil((float)n_T/BLOCK_SIZE), BLOCK_SIZE>>>(n_T,CheckSumHV_d,testLebel_d,testY_d,AccuVector_d,numTestSamples,numClasses);
	cudaDeviceSynchronize();
	
	
	
	//****************************************************
	//********** Testing : Accuracy Calculate ************
	//****************************************************
	

	cudaMemcpy(testLebel_h, testLebel_d, numTestSamples*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(AccuVector_h, AccuVector_d, numTestSamples*sizeof(int),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	for (int i=0;i<numTestSamples; i++)
	{
		Accuracy_d=Accuracy_d+AccuVector_h[i];
		//printf("\nTest Sample:%d;\tPredictedLabel=%d; \t Actual Label=%d\n",i,testLebel_h[i], testY_h[i]);
	}
	Accuracy_d=(Accuracy_d/numTestSamples)*100;

	
	//****************************************************
	//******** Free Dynamically Allocated Memory *********
	//****************************************************
	
	cudaFree(QueryHV_d);
	cudaFree(testQ_d);
	cudaFree(CheckSumHV_d);
	cudaFree(AccuVector_d);
	cudaFree(testLebel_d);
	
	free(AccuVector_h);
	free(testLebel_h);
	return Accuracy_d;
	
}










void Training_HV(float *trainX_d,int *trainY_d,float *validX_d, int *validY_d, float *L_d,int *trainQ_d,int numTrainSamples,int numValidSamples, int numFeatures,int numClasses,int M,int D, int *LD_d, int *ID_d, int *ClassHV_d, int *Classes_h, float ephsilon, int max_epoch, int min_epoch)
{
	
	//******************************************************
    //**Initialize thread block and kernel grid dimensions**
	//******************************************************
	
    const unsigned int BLOCK_SIZE = 512; 
	
	int n=numTrainSamples*numFeatures;
	int n_SHV=D*numTrainSamples;
	
	dim3 dimGrid(ceil(n/BLOCK_SIZE),1,1);
	dim3 dimBlock(BLOCK_SIZE,1,1);


	
	
	//******************************************************
    //*****Dynamic Memory Allocation in Device and Host*****
	//******************************************************
	
	int *Classes_d, *SampleHV_d,*ClassHVdec_d, *RetrainSumHV_d, *RetrainLebel_d,*AccuRetrain_d;
	cudaMalloc((void **)&Classes_d, numClasses*sizeof(int));
	cudaMalloc((void **)&SampleHV_d, numTrainSamples*D*sizeof(int));
	cudaMalloc((void **)&ClassHVdec_d, numClasses*D*sizeof(int));
	cudaMalloc((void **)&RetrainSumHV_d, numTrainSamples*numClasses*sizeof(int));
	cudaMalloc((void **)&RetrainLebel_d, numTrainSamples*sizeof(int));
	cudaMalloc((void **)&AccuRetrain_d, numTrainSamples*sizeof(int));
	
	
	
	
	//******************
    //***Quantization***
	//******************

	QuantKernel<<<ceil((float)n/BLOCK_SIZE), BLOCK_SIZE>>>(n,M,trainX_d,trainQ_d,L_d);
	cudaDeviceSynchronize();

	
	
	
	
	//*********************
    //***Sample Encoding***
	//*********************
	
	SampleHVKernel<<<ceil((float)n_SHV/BLOCK_SIZE), BLOCK_SIZE>>>(n_SHV,D,numTrainSamples, numFeatures,trainQ_d,SampleHV_d,LD_d, ID_d);
	cudaDeviceSynchronize();

	int *trainQ_h;
	trainQ_h=(int*)malloc(sizeof(int)*D*numTrainSamples);
	
	cudaMemcpy(trainQ_h,SampleHV_d,sizeof(int)*D*numTrainSamples,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	for (int i=0; i<25;i++)
	{
		printf("% d",trainQ_h[i]);
	}
	printf("\n\n\n");
	for (int i=0; i<25;i++)
	{
		printf(" %d",trainQ_h[numFeatures+i]);
	}
	printf("\n");
	

	
	
	//**********************************************
    //******Class Hyper Vector Majority Search******
	//**********************************************
	
	ClassHVAddKernel<<<ceil((float)n_SHV/BLOCK_SIZE), BLOCK_SIZE>>>(n_SHV,D,trainY_d,SampleHV_d,ClassHVdec_d,Classes_d);
	cudaDeviceSynchronize();
	cudaMemcpy(Classes_h, Classes_d, numClasses*sizeof(int),cudaMemcpyDeviceToHost);
	
	
	
	
	
	//**********************************************
    //************ Class Hyper Vector **************
	//**********************************************
	
	int ClassHV_sz=D*numClasses;
	ClassHVKernel<<<ceil((float)ClassHV_sz/BLOCK_SIZE), BLOCK_SIZE>>>(ClassHV_sz,D, numClasses, ClassHV_d,ClassHVdec_d,Classes_d);
	cudaDeviceSynchronize();
	printf("\nInitial Training Done..\n");
	
	
	//**********************************************
	//******************Retraining******************
	//**********************************************
	
	int epoch=1;
	float accuvalidinit=0.0;
	float accuvalid=0.0;
	
	accuvalid=TestingAccuracy(validX_d, validY_d, ClassHV_d, L_d, LD_d, ID_d,  D, M,  numValidSamples,  numFeatures, numClasses);
	
	while ((((accuvalid-accuvalidinit)>ephsilon)|(min_epoch>epoch))&(epoch<max_epoch))
	{
		printf("\nEpoch %d; \t Validation Accuracy = %f\n", epoch, accuvalid);
		accuvalidinit=accuvalid;
		
	
		CheckSumKernel<<<ceil((float)numTrainSamples*numClasses/BLOCK_SIZE), BLOCK_SIZE>>>(numTrainSamples*numClasses, SampleHV_d,ClassHV_d,RetrainSumHV_d,numTrainSamples,numClasses, D);
		cudaDeviceSynchronize();
		
		TestingKernel<<<ceil((float)numTrainSamples/BLOCK_SIZE), BLOCK_SIZE>>>(numTrainSamples,RetrainSumHV_d,RetrainLebel_d,trainY_d,AccuRetrain_d,numTrainSamples,numClasses);
		cudaDeviceSynchronize();
		
		RetraindKernel<<<ceil((float)n_SHV/BLOCK_SIZE), BLOCK_SIZE>>>(n_SHV,D,trainY_d,RetrainLebel_d, SampleHV_d,ClassHVdec_d,Classes_d);
		ClassHVKernel<<<ceil((float)ClassHV_sz/BLOCK_SIZE), BLOCK_SIZE>>>(ClassHV_sz,D, numClasses, ClassHV_d,ClassHVdec_d,Classes_d);
		cudaDeviceSynchronize();
		
		accuvalid=TestingAccuracy(validX_d, validY_d, ClassHV_d, L_d,LD_d,ID_d, D,M, numValidSamples, numFeatures,numClasses);
		epoch=epoch+1;
	}
	printf("\nFinal Training Done..\n");
	
	
	
	
	//***************************************
	//***Free dynamically allocated memory***
	//***************************************

	cudaFree(SampleHV_d);
	cudaFree(Classes_d);
	cudaFree(ClassHVdec_d);
	cudaFree(RetrainLebel_d);
	cudaFree(RetrainSumHV_d);
	cudaFree(AccuRetrain_d);
	
	printf("\nTraining Done...\n");
}



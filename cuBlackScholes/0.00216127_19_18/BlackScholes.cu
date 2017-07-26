/*
 ============================================================================
 Name        : blackscholes.cu
 Author      : Mungio
 Version     :
 Copyright   : CopiaBellaeBuona
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include "cuda_fp16.h"
using namespace std;


extern "C" void BlackScholesCPU(
    float *h_CallResult,
    float *h_PutResult,
    float *h_StockPrice,
    float *h_OptionStrike,
    float *h_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
);

///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
__device__ inline float cndGPU(float d)
{
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float
    K = 1.0f / (1.0f + __half2float(__hmul(__float2half(0.2316419f), __float2half(fabsf(d)))));

    float
    cnd = __half2float(__hmul(__float2half(RSQRT2PI) , __float2half(__expf(- 0.5f * d * d)))) * __half2float((__hmul(__float2half(K) , __float2half((A1 + K * (A2 + __half2float(__hmul(__float2half(K) , __float2half((A3 + __half2float(__hmul(__float2half(K) , (__hadd(__float2half(A4) , __float2half(K * A5)))))))))))))));

    if (d > 0)
        cnd = __half2float(__hsub(__float2half(1.0f) , __float2half(cnd)));

    return cnd; //Purtroppo il tipo di ritorno della funzione è float, questa cosa non ne teniamo conto, quindi questo cast pure lo pago.
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ inline void BlackScholesBodyGPU(
    float &CallResult,
    float &PutResult,
    float S, //Stock price
    float X, //Option strike
    float T, //Option years
    float R, //Riskless rate
    float V  //Volatility rate
)
{
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    float parrotInput[3];
    float parrotOutput[1];

    parrotInput[0] = S;
    parrotInput[1] = X;
    parrotInput[2] = T;

    sqrtT = sqrtf(T);
    d1 = __half2float(hdiv((__hadd(__float2half(__logf(S / X)) , (__hmul(__hadd(__float2half(R) , __hmul(__hmul(__float2half(0.5f) , __float2half(V)) , __float2half(V))) , __float2half(T))))) , (__hmul(__float2half(V) , __float2half(sqrtT)))));
    d2 = d1 - __half2float(__hmul(__float2half(V) , __float2half(sqrtT)));

    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);

    //Calculate Call and Put simultaneously
    expRT = __expf(- R * T);
    CallResult = S * CNDD1 - __half2float(__hmul(__hmul(__float2half(X) , __float2half(expRT)) , __float2half(CNDD2)));
    parrotOutput[0] = CallResult / 10.0;

    CallResult = parrotOutput[0] * 10.0;
    PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}


////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void BlackScholesGPU(
    float *d_CallResult,
    float *d_PutResult,
    float *d_StockPrice,
    float *d_OptionStrike,
    float *d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
)
{

    const int opt = blockDim.x * blockIdx.x + threadIdx.x;
    if (opt < optN)
        BlackScholesBodyGPU(
            d_CallResult[opt],
            d_PutResult[opt],
            d_StockPrice[opt],
            d_OptionStrike[opt],
            d_OptionYears[opt],
            Riskfree,
            Volatility
        );
}

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int OPT_N = 4000000;
const int  NUM_ITERATIONS = 1; // Amir: Change number of iteration


const int          OPT_SZ = OPT_N * sizeof(float);
const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	std::cout<<"Test"<<std::endl;
	std::cout.flush();
//#pragma parrot.start("BlackScholesBodyGPU")
    //'h_' prefix - CPU (host) memory space
    float
    //Results calculated by CPU for reference
    *h_CallResultCPU,
    *h_PutResultCPU,
    //CPU copy of GPU results
    *h_CallResultGPU,
    *h_PutResultGPU,
    //CPU instance of input data
    *h_StockPrice,
    *h_OptionStrike,
    *h_OptionYears;

    //'d_' prefix - GPU (device) memory space
    float
    //Results calculated by GPU
    *d_CallResult,
    *d_PutResult,
    //GPU instance of input data
    *d_StockPrice,
    *d_OptionStrike,
    *d_OptionYears;

    int i;

    h_CallResultCPU = (float *)malloc(OPT_SZ);
    h_PutResultCPU  = (float *)malloc(OPT_SZ);
    h_CallResultGPU = (float *)malloc(OPT_SZ);
    h_PutResultGPU  = (float *)malloc(OPT_SZ);
    h_StockPrice    = (float *)malloc(OPT_SZ);
    h_OptionStrike  = (float *)malloc(OPT_SZ);
    h_OptionYears   = (float *)malloc(OPT_SZ);

    cudaMalloc((void **)&d_CallResult,   OPT_SZ);
    cudaMalloc((void **)&d_PutResult,    OPT_SZ);
    cudaMalloc((void **)&d_StockPrice,   OPT_SZ);
    cudaMalloc((void **)&d_OptionStrike, OPT_SZ);
    cudaMalloc((void **)&d_OptionYears,  OPT_SZ);

    srand(5347);

    // Amir
    std::ifstream dataFile(argv[1]);
    int numberOptions;
    dataFile >> numberOptions;
    float stockPrice, optionStrike, optionYear;
    // Rima

    //Generate options set
    for (i = 0; i < numberOptions; i++)
    {
        h_CallResultCPU[i] = 0.0f;
        h_PutResultCPU[i]  = -1.0f;

        // Amir
        dataFile >> stockPrice >> optionStrike >> optionYear;
        h_StockPrice[i] = stockPrice;
        h_OptionStrike[i] = optionStrike;
        h_OptionYears[i] =  optionYear;
        // Rima
    }

    int optionSize = numberOptions * sizeof(float);

    //Copy options data to GPU memory for further processing
    cudaMemcpy(d_StockPrice,  h_StockPrice,   optionSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionStrike, h_OptionStrike,  optionSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionYears,  h_OptionYears,   optionSize, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();


    for (i = 0; i < NUM_ITERATIONS; i++)
    {
        BlackScholesGPU<<<DIV_UP(numberOptions, 128), 128/*480, 128*/>>>(
            d_CallResult,
            d_PutResult,
            d_StockPrice,
            d_OptionStrike,
            d_OptionYears,
            RISKFREE,
            VOLATILITY,
            numberOptions
        );

    }

    cudaDeviceSynchronize();

    //Read back GPU results to compare them to CPU results
    cudaMemcpy(h_CallResultGPU, d_CallResult, optionSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_PutResultGPU,  d_PutResult,  optionSize, cudaMemcpyDeviceToHost);

    // Amir
    ofstream callResultFile;
    callResultFile.open(argv[2]);
    for (i = 0 ; i < numberOptions; i++)
    {
        callResultFile << h_CallResultGPU[i] << std::endl;
    }
    callResultFile.close();
    // Rima


//#pragma parrot.end("BlackScholesBodyGPU")

    cudaFree(d_OptionYears);
    cudaFree(d_OptionStrike);
    cudaFree(d_StockPrice);
    cudaFree(d_PutResult);
    cudaFree(d_CallResult);
    free(h_OptionYears);
    free(h_OptionStrike);
    free(h_StockPrice);
    free(h_PutResultGPU);
    free(h_CallResultGPU);
    free(h_PutResultCPU);
    free(h_CallResultCPU);

    cudaDeviceReset();

    //printf("Test passed\n");
    exit(EXIT_SUCCESS);
}


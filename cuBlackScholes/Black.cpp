// Amir
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>

#include "helperIntrinsics.h"

using namespace std;

#include "vpa.h"

typedef struct threadIdx_struct{int x; int y; int z;} threadIdx_t;
typedef struct blockDim_struct{int x; int y; int z;} blockDim_t;
typedef struct blockIdx_struct{int x; int y; int z;} blockIdx_t;

threadIdx_t threadIdx = {0,0,0};
blockDim_t blockDim = {0,0,0};
blockIdx_t blockIdx = {0,0,0};

::vpa::FloatingPointPrecision OP_14 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_13 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_12 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_11 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_10 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_9 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_8 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_7 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_6 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_5 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_4 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_3 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_2 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_1 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_0 = ::vpa::float_prec;
float cndGPU(float d)
{
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float K = (float)(::vpa::VPA(1.0f , OP_0) /::vpa::VPA( (::vpa::VPA(1.0f , OP_1) +::vpa::VPA( ::vpa::VPA(0.2316419f , OP_2)* ::vpa::VPA(fabsf(d), OP_2), OP_1)/*II OP_1*/ ), OP_0)/*II OP_0*/ ) ;

    float cnd = (float)(::vpa::VPA(::vpa::VPA(RSQRT2PI , OP_4)* ::vpa::VPA(__expf(- 0.5f * d * d), OP_4) , OP_3)*
          ::vpa::VPA((::vpa::VPA(K , OP_5) *::vpa::VPA( (::vpa::VPA(A1 , OP_6) +::vpa::VPA( ::vpa::VPA(K , OP_7) *::vpa::VPA( (::vpa::VPA(A2 , OP_8) +::vpa::VPA( ::vpa::VPA(K , OP_9) *::vpa::VPA( (::vpa::VPA(A3 , OP_10) +::vpa::VPA( ::vpa::VPA(K , OP_11) *::vpa::VPA( (::vpa::VPA(A4 , OP_12) +::vpa::VPA( ::vpa::VPA(K , OP_13)* ::vpa::VPA(A5, OP_13), OP_12)/*II OP_12*/ ), OP_11)/*II OP_11*/ , OP_10)/*II OP_10*/ ), OP_9)/*II OP_9*/ , OP_8)/*II OP_8*/ ), OP_7)/*II OP_7*/ , OP_6)/*II OP_6*/ ), OP_5)/*II OP_5*/ ), OP_3)) ;

    if (d > 0)
        cnd =(float)( ::vpa::VPA(1.0f , OP_14)- ::vpa::VPA(cnd, OP_14));

    return cnd;
}


::vpa::FloatingPointPrecision OP_33 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_32 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_31 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_30 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_29 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_28 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_27 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_26 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_25 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_24 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_23 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_22 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_21 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_20 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_19 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_18 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_17 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_16 = ::vpa::float_prec;
::vpa::FloatingPointPrecision OP_15 = ::vpa::float_prec;
void BlackScholesBodyGPU(
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
    
    sqrtT = sqrtf(T);
    d1 =(float)( ::vpa::VPA((::vpa::VPA(__logf(S / X) , OP_16) +::vpa::VPA( ::vpa::VPA((::vpa::VPA(R , OP_18) +::vpa::VPA( ::vpa::VPA(::vpa::VPA(0.5f , OP_20)* ::vpa::VPA(V, OP_20) , OP_19) *::vpa::VPA( V, OP_19)/*II OP_19*/ , OP_18)/*II OP_18*/ ) , OP_17) *::vpa::VPA( T, OP_17)/*II OP_17*/ , OP_16)/*II OP_16*/ ) , OP_15)/ ::vpa::VPA((::vpa::VPA(V , OP_21)* ::vpa::VPA(sqrtT, OP_21)), OP_15));
    d2 =(float)( ::vpa::VPA(d1 , OP_22) -::vpa::VPA( ::vpa::VPA(V , OP_23)* ::vpa::VPA(sqrtT, OP_23), OP_22)/*II OP_22*/ );

    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);

    //Calculate Call and Put simultaneously
    expRT = __expf(- R * T);
    CallResult =(float)( ::vpa::VPA(::vpa::VPA(S , OP_25)* ::vpa::VPA(CNDD1, OP_25) , OP_24)- ::vpa::VPA(::vpa::VPA(::vpa::VPA(X , OP_27)* ::vpa::VPA(expRT, OP_27) , OP_26) *::vpa::VPA( CNDD2, OP_26)/*II OP_26*/ , OP_24));
   
    PutResult  =(float)( ::vpa::VPA(::vpa::VPA(::vpa::VPA(X , OP_30)* ::vpa::VPA(expRT, OP_30) , OP_29)* ::vpa::VPA((::vpa::VPA(1.0f , OP_31)- ::vpa::VPA(CNDD2, OP_31)), OP_29) , OP_28)- ::vpa::VPA(::vpa::VPA(S , OP_32) *::vpa::VPA( (::vpa::VPA(1.0f , OP_33)- ::vpa::VPA(CNDD1, OP_33)), OP_32)/*II OP_32*/ , OP_28));
}


void BlackScholesGPU(
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


int main(int argc, char* argv[]){
    if(argc < 2){
        std::cerr << "you must provide a test input file" << std::endl;
        return -1;
    }
    ::std::ofstream oracle ( "bs_oracle.txt", ::std::ofstream::out );
    ::std::ifstream dataFile ( argv[1], ::std::ofstream::out );
    int numberOptions;
    dataFile >> numberOptions;
    float h_StockPrice[1], h_OptionStrike[1], h_OptionYears[1], callResult[1], putResult[1];

    //Generate options set
    for (int i = 0; i < numberOptions; i++)
    {
        dataFile >> h_StockPrice[0] >> h_OptionStrike[0] >> h_OptionYears[0];
        BlackScholesGPU(callResult, putResult, h_StockPrice, h_OptionStrike, h_OptionYears, RISKFREE, VOLATILITY, numberOptions);
        oracle << ::std::setprecision ( ::std::numeric_limits<double>::digits10 + 1 ) << h_StockPrice[0] << " " << h_OptionStrike[0] << " " <<  h_OptionYears[0] << " "
               << ::std::setprecision ( ::std::numeric_limits<double>::digits10 + 1 )
               << callResult[0] << " " << putResult[0] << "\n";
    }
    return 0;

}


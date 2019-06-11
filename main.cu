#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <cassert>
#include <cudnn.h>

#define CHECK(call)                                                  \
{                                                                    \
    const cudaError_t error = call;                                  \
    std::cout << "CHECK cudaError_t: ";                              \
    if (error != cudaSuccess)                                        \
    {                                                                \
        std::cout << __FILE__                                        \
                  << "("                                             \
                  << __LINE__                                        \
                  << ")"                                             \
                  << ": "                                            \
                  << "Error"                                         \
                  << std::endl;                                      \
        std::cout << "code: "                                        \
                  << error                                           \
                  << ", "                                            \
                  << "reason: "                                      \
                  << cudaGetErrorString(error)                       \
                  << std::endl;                                      \
        std::exit(EXIT_FAILURE);                                     \
    }                                                                \
    else                                                             \
    {                                                                \
        std::cout << __FILE__                                        \
                  << "("                                             \
                  << __LINE__                                        \
                  << ")"                                             \
                  << ": "                                            \
                  << "cudaSuccess"                                   \
                  << std::endl;                                      \
    }                                                                \
}

#define checkCUDNN(call)                                             \
{                                                                    \
    cudnnStatus_t status = (call);                                   \
    if (status != CUDNN_STATUS_SUCCESS) {                            \
        std::cout << __FILE__                                        \
                  << "("                                             \
                  << __LINE__                                        \
                  << ")"                                             \
                  << ": "                                            \
                  << "Error"                                         \
                  << std::endl;                                      \
        std::cout << "code: "                                        \
                  << status                                          \
                  << ", "                                            \
                  << "reason: "                                      \
                  << cudnnGetErrorString(status)                     \
                  << std::endl;                                      \
        std::exit(EXIT_FAILURE);                                     \
    }                                                                \
}

void rand_vector_float (std::vector<float> &v);
namespace {
    const char* getAlgoName(cudnnConvolutionBwdFilterAlgo_t algo);
}

int main(int argc, char *argv[]) {
    cudnnHandle_t handle;
    checkCUDNN(cudnnCreate(&handle));
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    const int n = 5, ci = 4, hi = 1, wi = 1, 
          pad_h = 1, pad_w = 1, u = 1, v = 1, 
          dilation_h = 1, dilation_w = 1,
          co = 3, kernel_h = 1, kernel_w = 1;
    int n_dmy, co_dmy, ho, wo;

    cudnnConvolutionDescriptor_t convDesc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(
                /*cudnnConvolutionDescriptor_t*/ convDesc,
                /*int                         */ pad_h,
                /*int                         */ pad_w,
                /*int                         */ u,
                /*int                         */ v,
                /*int                         */ dilation_h,
                /*int                         */ dilation_w,
                /*cudnnConvolutionMode_t      */ CUDNN_CROSS_CORRELATION,
                /*cudnnDataType_t             */ CUDNN_DATA_FLOAT));

    cudnnTensorDescriptor_t xDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    checkCUDNN(cudnnSetTensor4dDescriptor(
                /*cudnnTensorDescriptor_t*/ xDesc,
                /*cudnnTensorFormat_t    */ CUDNN_TENSOR_NCHW,
                /*cudnnDataType_t        */ CUDNN_DATA_FLOAT,
                /*int                    */ n,
                /*int                    */ ci,
                /*int                    */ hi,
                /*int                    */ wi));


    cudnnFilterDescriptor_t dwDesc;
    checkCUDNN(cudnnCreateFilterDescriptor(&dwDesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(
                /*cudnnFilterDescriptor_t*/ dwDesc,
                /*cudnnDataType_t        */ CUDNN_DATA_FLOAT,
                /*cudnnTensorFormat_t    */ CUDNN_TENSOR_NCHW,
                /*int                    */ co,
                /*int                    */ ci,
                /*int                    */ kernel_h,
                /*int                    */ kernel_w));

    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
                /*const cudnnConvolutionDescriptor_t*/ convDesc,
                /*const cudnnTensorDescriptor_t     */ xDesc,
                /*const cudnnFilterDescriptor_t     */ dwDesc,
                /*int*                              */ &n_dmy,
                /*int*                              */ &co_dmy,
                /*int*                              */ &ho,
                /*int*                              */ &wo));
    assert(n == n_dmy);
    assert(co == co_dmy);

    cudnnTensorDescriptor_t dyDesc;
    cudnnCreateTensorDescriptor(&dyDesc);
    checkCUDNN(cudnnSetTensor4dDescriptor(
                /*cudnnTensorDescriptor_t*/ dyDesc,
                /*cudnnTensorFormat_t    */ CUDNN_TENSOR_NCHW,
                /*cudnnDataType_t        */ CUDNN_DATA_FLOAT,
                /*int                    */ n,
                /*int                    */ co,
                /*int                    */ ho,
                /*int                    */ wo));

    cudnnConvolutionBwdFilterAlgo_t algo;
    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
                /*cudnnHandle_t                        */ handle,
                /*const cudnnTensorDescriptor_t        */ xDesc,
                /*const cudnnTensorDescriptor_t        */ dyDesc,
                /*const cudnnConvolutionDescriptor_t   */ convDesc,
                /*const cudnnFilterDescriptor_t        */ dwDesc,
                /*cudnnConvolutionBwdFilterPreference_t*/ CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                /*size_t (is not used)                 */ 0,
                /*cudnnConvolutionBwdFilterAlgo_t      */ &algo));
    std::cout << "cudnnConvolutionBwdFilterAlgo_t: " << getAlgoName(algo) << std::endl;

    size_t workSpaceSizeInBytes;
    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                /*cudnnHandle_t                     */ handle,
                /*const cudnnTensorDescriptor_t     */ xDesc,
                /*const cudnnTensorDescriptor_t     */ dyDesc,
                /*const cudnnConvolutionDescriptor_t*/ convDesc,
                /*const cudnnFilterDescriptor_t     */ dwDesc,
                /*cudnnConvolutionBwdFilterAlgo_t   */ algo,
                /*size_t                            */ &workSpaceSizeInBytes));

    std::vector<float> h_x(n * ci * hi * wi, 0);
    std::vector<float> h_dy(n * co * ho * wo, 0);
    std::vector<float> h_dw(co * ci * kernel_h * kernel_w,
            std::numeric_limits<float>::quiet_NaN());
    std::vector<float> h_dw_expct(co * ci * kernel_h * kernel_w,
            std::numeric_limits<float>::quiet_NaN());
    rand_vector_float(h_x);
    rand_vector_float(h_dy);

    void *x = nullptr, *dy = nullptr, *dw = nullptr, *workSpace = nullptr;
    size_t size_x = n * ci * hi * wi * sizeof(float);
    size_t size_dy = n * co * ho * wo * sizeof(float);
    size_t size_dw = co * ci * kernel_h * kernel_w * sizeof(float);
    cudaMalloc (&x, size_x);
    cudaMalloc (&dy, size_dy);
    cudaMalloc (&dw, size_dw);
    cudaMalloc (&workSpace, workSpaceSizeInBytes);
    cudaMemset (dw, 0xff, size_dw);

    cudaMemcpy(x, h_x.data(), size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, h_dy.data(), size_dy, cudaMemcpyHostToDevice);

    const float alpha = 1, beta = 0;
    cudaEventRecord(start);
    checkCUDNN(cudnnConvolutionBackwardFilter(
                /*cudnnHandle_t                     */ handle,
                /*const void *                      */ &alpha,
                /*const cudnnTensorDescriptor_t     */ xDesc,
                /*const void *                      */ x,
                /*const cudnnTensorDescriptor_t     */ dyDesc,
                /*const void *                      */ dy,
                /*const cudnnConvolutionDescriptor_t*/ convDesc,
                /*cudnnConvolutionBwdFilterAlgo_t   */ algo,
                /*void *                            */ workSpace,
                /*size_t                            */ workSpaceSizeInBytes,
                /*const void *                      */ &beta,
                /*const cudnnFilterDescriptor_t     */ dwDesc,
                /*void *                            */ dw));
    cudaEventRecord(stop);

    CHECK(cudaDeviceSynchronize());
    cudaMemcpy(h_dw.data(), dw, size_dw, cudaMemcpyDeviceToHost);

    for (std::vector<float>::const_iterator i = h_dw.begin(); i != h_dw.end(); ++i)
        std::cout << *i << ' ';
    std::cout << std::endl;

    for (std::vector<float>::const_iterator i = h_dw_expct.begin(); i != h_dw_expct.end(); ++i)
        std::cout << *i << ' ';
    std::cout << std::endl;

    float msec = 0;
    cudaEventElapsedTime(&msec, start, stop);
    std::cout << "Exec time: " << msec * 1000 << "[usec]" << std::endl;

    cudaFree(x);
    cudaFree(dy);
    cudaFree(dw);
    cudaFree(workSpace);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(dyDesc);
    cudnnDestroyFilterDescriptor(dwDesc);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudnnDestroy(handle);
    CHECK(cudaDeviceSynchronize());
    return 0;
}

std::mt19937 mt(0);
void rand_vector_float (std::vector<float> &v) {
    std::normal_distribution<> rand(0, 5);
    for (std::vector<float>::iterator i = v.begin(); i != v.end(); ++i) {
        *i = rand(mt);
    }
    return;
}

namespace {
    const char* getAlgoName(cudnnConvolutionBwdFilterAlgo_t algo) 
    {
        switch (algo) 
        {
            case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0: return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0"; /* non-deterministic */
            case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1: return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1";
            case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT: return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT";
            case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3: return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3"; /* non-deterministic */
            case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD: return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD"; /* not implemented */
            case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED: return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED";
            case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING: return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING";
            case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT: return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT";
            default: std::exit(EXIT_FAILURE); return "Error";
        }
    }
}

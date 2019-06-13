/**
 * Example usage of cuDNN convolution backward filter
 * CUDA Version: 10.0
 * cuDNN Version: 7.4
 *
 * @author     ryotakata
 **/
#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <cassert>
#include <cudnn.h>

#define CHECK(call)                                                  \
{                                                                    \
    const cudaError_t error = call;                                  \
    if (error != cudaSuccess)                                        \
    {                                                                \
        std::cout << "CHECK cudaError_t: "                           \
                  << __FILE__                                        \
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
void pseudoConvolutionBackwardFilter(
        const std::vector<float> &x, 
        const std::vector<float> &dy,
        std::vector<float> &dw,
        const int N, const int  Ci, const int Hi, const int Wi,
        const int Co, const int Ho, const int Wo,
        const int Hk, const int Wk, const int Hs, const int Ws,
        const int Hp, const int Wp);
float getMaxAbsError(const std::vector<float> &exp, const std::vector<float> &act);

int main(int argc, char *argv[]) {
    cudnnHandle_t handle;
    checkCUDNN(cudnnCreate(&handle));
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    const int n = 32, ci = 2, hi = 14, wi = 14, 
          pad_h = 0, pad_w = 0, u = 1, v = 1, 
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

    float msec = 0;
    cudaEventElapsedTime(&msec, start, stop);
    std::cout << "Exec time: " << msec * 1000 << "[usec]" << std::endl;

    pseudoConvolutionBackwardFilter(h_x, h_dy, h_dw_expct,
            n, ci, hi, wi,
            co, ho, wo,
            kernel_h, kernel_w, u, v,
            pad_h, pad_w);
    cudaMemcpy(h_dw.data(), dw, size_dw, cudaMemcpyDeviceToHost);
    std::cout << "Max Abs Error(expect vs actual): "
              << getMaxAbsError(h_dw_expct, h_dw) << std::endl;



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

namespace {
    int getIndex(
            const int n,
            const int c,
            const int h,
            const int w,
            const int N,
            const int C,
            const int H,
            const int W
            ) {
        return n * C * H * W
            + c * H * W
            + h * W
            + w;
    }
}

void pseudoConvolutionBackwardFilter(
        const std::vector<float> &x, 
        const std::vector<float> &dy,
        std::vector<float> &dw,
        const int N, const int  Ci, const int Hi, const int Wi,
        const int Co, const int Ho, const int Wo,
        const int Hk, const int Wk, const int Hs, const int Ws,
        const int Hp, const int Wp
        ) {

    for (std::vector<float>::iterator i = dw.begin(); i != dw.end(); ++i)
        *i = 0.f;

    int idx_x, idx_dy, idx_dw;
    for (int hi = 0; hi < Hi; ++hi) {
        for (int ho = 0; ho < Ho; ++ho) {
            for (int hk = 0; hk < Hk; ++hk) {
                if ((ho * Hs + hk) != (hi + Hp)) {
                    continue;
                }
                for (int wi = 0; wi < Wi; ++wi) {
                    for (int wo = 0; wo < Wo; ++wo) {
                        for (int wk = 0; wk < Wk; ++wk) {
                            if ( (wo * Ws + wk) != (wi + Wp)) {
                                continue;
                            }
                            for (int n = 0; n < N; ++n) {
                                for (int ci = 0; ci < Ci; ++ci) {
                                    for (int co = 0; co < Co; ++co) {
                                        idx_x  = getIndex(n, ci, hi, wi, N, Ci, Hi, Wi);
                                        idx_dy = getIndex(n, co, ho, wo, N, Co, Ho, Wo);
                                        idx_dw = getIndex(co, ci, hk, wk, Co, Ci, Hk, Wk);
                                        dw[idx_dw] += x[idx_x] * dy[idx_dy];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return;
}

float getMaxAbsError(const std::vector<float> &exp, const std::vector<float> &act) {
    float tmp, ret;
    std::vector<float> abs_err(exp.size(),0.f);
    for (std::vector<float>::iterator i = abs_err.begin(); i != abs_err.end(); ++i) {
        size_t index = std::distance(abs_err.begin(), i);
        tmp  = act[index] - exp[index];
        *i = (tmp >= 0) ? tmp : -tmp;
    }

    ret = *std::max_element(abs_err.begin(), abs_err.end());
    return ret;
}

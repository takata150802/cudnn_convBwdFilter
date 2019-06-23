#include <iostream>
#include "prof_conv_bwd_filter.h"

int main (void) {

    float msec;
    int   max_ulp;
    std::string algo_name;
    int   flop, byte;
    profConvBwdFilter(
            /*const int n,*/               1,
            /*const int ci,*/              1,
            /*const int hi,*/              1,
            /*const int wi,*/              1,
            /*const int co,*/              1,
            /*const int u,*/               1,
            /*const int v,*/               1,
            /*const int kernel_h,*/        1,
            /*const int kernel_w,*/        1,
            /*const int pad_h,*/           0,
            /*const int pad_w,*/           0,
            /*const int dilation_h,*/      1,
            /*const int dilation_w,*/      1,
            /*float& msec,*/            msec,
            /*int& max_ulp*/         max_ulp,
            /*std::string*/        algo_name,
            /*int&*/                    flop,
            /*int&*/                    byte
            );
   std::cout << __FILE__ << ": "
             << "Exec time: " 
             << msec * 1000 << "[usec]" 
             << std::endl;
   std::cout << __FILE__ << ": "
             << "Max Ulp Error(expect vs actual): "
             << max_ulp 
             << std::endl;
   std::cout << __FILE__ << ": "
             << "cudnnConvolutionBwdFilterAlgo_t: "
             << algo_name
             << std::endl;
   std::cout << __FILE__ << ": "
             << "Arithmetic Intensity: "
             << float(flop) /  float(byte)
             << std::endl;
   std::cout << __FILE__ << ": "
             << float(flop) << " " <<  float(byte)
             << std::endl;
}

#include <iostream>
#include "prof_conv_bwd_filter.h"

int main (void) {

    float msec;
    int   max_ulp;
    profConvBwdFilter(
            /*const int n,*/               1,
            /*const int ci,*/              2,
            /*const int hi,*/              3,
            /*const int wi,*/              4,
            /*const int co,*/              5,
            /*const int u,*/               1,
            /*const int v,*/               2,
            /*const int kernel_h,*/        3,
            /*const int kernel_w,*/        4,
            /*const int pad_h,*/           1,
            /*const int pad_w,*/           2,
            /*const int dilation_h,*/      1,
            /*const int dilation_w,*/      1,
            /*float& msec,*/            msec,
            /*int& max_ulp*/         max_ulp
            );
   std::cout << __FILE__ << ": "
             << "Exec time: " 
             << msec * 1000 << "[usec]" 
             << std::endl;
   std::cout << __FILE__ << ": "
             << "Max Ulp Error(expect vs actual): "
             << max_ulp 
             << std::endl;
}

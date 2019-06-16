#ifndef RUN_CONV_BWD_FILTER_H
#define RUN_CONV_BWD_FILTER_H

int runConvBwdFilter(
        const int n,
        const int ci,
        const int hi,
        const int wi,
        const int co,
        const int u,
        const int v,
        const int kernel_h,
        const int kernel_w,
        const int pad_h,
        const int pad_w,
        const int dilation_h,
        const int dilation_w
        );
#endif

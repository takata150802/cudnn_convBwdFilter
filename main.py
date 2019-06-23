#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 19:05:29 2019
@author: ryotakata
"""

"""setting parameters"""
"""machine spec"""
MACHINE_NAME = "GTX 1060 3 GB"
# * Base Clock 1506MHz
# * Boost Clock 1708MHz
#      #cuda core * (add+mul) *           freq * [Gflop] -> 3,469[Gflops]
PEAK_PERF =  1152 *         2 * (1506 * 1e+6) * 1e-9 
#   #mem_chips * Bus_width * Gbps * [GB/s] -> 192[GB/s]
BAND_WIDTH = 6 *        32 *    8 * 1/8
MACHINE_SPEC = MACHINE_NAME + "," \
               + str(int(PEAK_PERF)) + "Gflops," \
               + str(int(BAND_WIDTH)) + "GB/s"

import pandas as pd
import matplotlib.pyplot as plt

def main():

    """profile cudnn Convoultion Backward Filter"""
    df = prof_conv_bwd_filter()
    df.to_csv("prof.cudnnConvBwdFilter.csv")

    """visualization, Roofline model"""
    df = pd.read_csv('prof.cudnnConvBwdFilter.csv', header=0, index_col=0)
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1, 1, 1)
    plot_rooline (ax, MACHINE_SPEC, PEAK_PERF, BAND_WIDTH)
    plot_result (ax, df)
    # fig.subplots_adjust(right=0.8)
    plt.subplots_adjust(left=0.1, right=0.6)
    plt.savefig('roofline.png')
    return

def prof_conv_bwd_filter ():
    from cudnn import convBwdFilter
    from chainer.utils import get_conv_outsize
    from random import seed
    from random import randint

    df = pd.DataFrame(columns=[
            'msec',
            'max_ulp',
            'algo_name', 
            'flop', 
            'byte',
            'arithmetic_intensity',
            'GFlops',
            ])

    count = 0
    seed(0)
    while (count < 1000):
        n = randint(1, 32)
        ci = randint(1, 32)
        hi = randint(1, 122)
        wi = randint(1, 122)
        co = randint(1, 32)
        u = randint(1, 7)
        v = randint(1, 7)
        kernel_h = randint(1, 40)
        kernel_w = randint(1, 40)
        pad_h = randint(0, 8)
        pad_w = randint(0, 8)
        dilation_h = 1
        dilation_w = 1
        assert dilation_w == 1
        assert dilation_h == 1
        ho = get_conv_outsize(hi, u, kernel_h, pad_h, cover_all=False, d=dilation_h)
        wo = get_conv_outsize(wi, v, kernel_w, pad_w, cover_all=False, d=dilation_w)
        if ho <= 0 or wo <= 0:
            continue
        if (pad_h + hi ) < kernel_h:
            continue
        if (pad_w + wi ) < kernel_w:
            continue
        ret, msec, max_ulp, algo_name, flop, byte = convBwdFilter(
            n,
            ci,
            hi,
            wi,
            co,
            u,
            v,
            kernel_h,
            kernel_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w
            )
        assert ret == 0
        ai = float(flop) / byte # [flop/Byte]
        flops = float(flop) / (msec * 1e-3 ) 
        gf = flops * 1e-9
        ll = [msec, max_ulp, algo_name, flop, byte, ai, gf]
        ss = pd.Series(ll, index=df.columns )
        df = df.append(ss, ignore_index=True )
        count += 1
    # print (df)
    return df

def plot_rooline (ax, MACHINE_SPEC, PEAK_PERF, BAND_WIDTH):
    import numpy as np
    x = np.arange(1e-2, 1e+3, 1e-2)
    left_roof = x * BAND_WIDTH
    temp = []
    for i in range(len(x)):
        temp.append(min(left_roof[i],PEAK_PERF))
    y = np.array(temp)
    ax = ax.plot(x,y)
    return ax

def plot_result (ax, df):
    import pandas as pd
    ls_legend = [MACHINE_SPEC,]
    dd = df[df.algo_name == 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0']
    if len(dd) > 0:
        dd.plot(ax=ax, kind='line', x='arithmetic_intensity', y='GFlops', style=['b.'], alpha=0.5)
        ls_legend.append('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0')
    dd = df[df.algo_name == 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1']
    if len(dd) > 0:
        dd.plot(ax=ax, kind='line', x='arithmetic_intensity', y='GFlops', style=['g.'], alpha=0.5)
        ls_legend.append('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1')
    dd = df[df.algo_name == 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT']
    if len(dd) > 0:
        dd.plot(ax=ax, kind='line', x='arithmetic_intensity', y='GFlops', style=['r.'], alpha=0.5)
        ls_legend.append('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT')
    dd = df[df.algo_name == 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3']
    if len(dd) > 0:
        dd.plot(ax=ax, kind='line', x='arithmetic_intensity', y='GFlops', style=['c.'], alpha=0.5)
        ls_legend.append('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3')
    dd = df[df.algo_name == 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED']
    if len(dd) > 0:
        dd.plot(ax=ax, kind='line', x='arithmetic_intensity', y='GFlops', style=['m.'], alpha=0.5)
        ls_legend.append('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED')
    dd = df[df.algo_name == 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING']
    if len(dd) > 0:
        dd.plot(ax=ax, kind='line', x='arithmetic_intensity', y='GFlops', style=['y.'], alpha=0.5)
        ls_legend.append('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING')
    dd = df[df.algo_name == 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT']
    if len(dd) > 0:
        dd.plot(ax=ax, kind='line', x='arithmetic_intensity', y='GFlops', style=['k.'], alpha=0.5)
        ls_legend.append('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT')

    ax.set_ylabel('Performance [Gflops]')
    ax.legend(ls_legend, bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(which="both")
    return ax

def argsparse():
    import sys
    args = sys.argv
    assert len(args) >= 1,\
    "m(-_-)m This script takes no arguments.\n"
    return args[0]

if __name__ == '__main__':
    """
    """
    this_script_name = argsparse()
    main()

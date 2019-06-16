import cython

def func():
    cdef:
        int ret
    ret = runConvBwdFilter(1,2,3,4,5,1,2,3,4,1,2,1,1)
    return ret

#     z =runConvBwdFilter( \
#             """const int n,"""               1, \
#             """const int ci,"""              2, \
#             """const int hi,"""              3, \
#             """const int wi,"""              4, \
#             """const int co,"""              5, \
#             """const int u,"""               1, \
#             """const int v,"""               2, \
#             """const int kernel_h,"""        3, \
#             """const int kernel_w,"""        4, \
#             """const int pad_h,"""           1, \
#             """const int pad_w,"""           2, \
#             """const int dilation_h,"""      1, \
#             """const int dilation_w"""       1 \
#             )

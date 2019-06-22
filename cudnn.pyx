import cython

def convBwdFilter(
        const int n, const int ci, const int hi, const int wi,
        const int co,
        const int u, const int v,
        const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int dilation_h, const int dilation_w):
    cdef:
        int ret
    ret =runConvBwdFilter( \
        n, \
        ci, \
        hi, \
        wi, \
        co, \
        u, \
        v, \
        kernel_h, \
        kernel_w, \
        pad_h, \
        pad_w, \
        dilation_h, \
        dilation_w)
    return ret

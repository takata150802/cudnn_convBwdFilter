import cython
def convBwdFilter(
        const int n, const int ci, const int hi, const int wi,
        const int co,
        const int u, const int v,
        const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int dilation_h, const int dilation_w):
    cdef:
        int ret = -1
        float msec = -1
        int max_ulp = -1
        string algo_name = ""
        int flop = -1;
        int byte = -1;
    ret =profConvBwdFilter(
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
        dilation_w,
        msec,
        max_ulp,
        algo_name,
        flop,
        byte)
    return ret, msec, max_ulp, algo_name.decode('UTF-8'), flop, byte

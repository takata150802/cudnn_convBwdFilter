from cudnn import convBwdFilter

n = 1
ci = 1
hi = 210
wi = 94
co = 1
u = 1
v = 1
kernel_h = 1
kernel_w = 2
pad_h = 0
pad_w = 0
dilation_h = 1
dilation_w = 1

convBwdFilter(\
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
    dilation_w  \
    )

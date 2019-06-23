from cudnn import convBwdFilter

n = 1
ci = 1
hi = 210
wi = 94
co = 1
u = 1
v = 1
kernel_h = 3
kernel_w = 3
pad_h = 0
pad_w = 0
dilation_h = 1
dilation_w = 1

for wi in range(3,30):
    for hi in range(3,30):
        ret, msec, max_ulp, algo_name = convBwdFilter(\
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
    
print (ret)
print (msec)
print (max_ulp)
print (algo_name)

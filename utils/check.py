import numpy as np
import jax.numpy as jnp

# find the result from the start location of signature file
def from_txt(fpath, golden, start ):
    ebyte = golden.itemsize
    size = golden.size
    dtype = golden.dtype
    # we need to save the result align ebyte so we need align_size here
    align_size = ebyte

    if golden.dtype == np.bool_:
        # set ebyte to 0.1 to be different with e8
        ebyte = 0.1
        align_size = 1

    result = []
    now = 0
    # we recompute the start location in order to satisfy the align mechanism
    if start % align_size != 0:
        start = ( start // align_size + 1 ) * align_size
    with open(fpath) as file:
        for line in file:
            line = line.rstrip()
            if start - now >= 32:
                # if start bigger than this line, continue to next line
                now += 32
                continue
            else:  
                # if start is in this line, we use start-now as line_start
                # if not we use 0 as line_start                
                line_start = start - now
                if line_start < 0:
                    line_start = 0    

                if ebyte != 0.1:
                    # handle e8\e16\e32\e64
                    while line_start != 32:
                        # we get hex string from end to start because they are saved in that way
                        if line_start == 0:
                            str = line[-2*ebyte:]
                        else:
                            str = line[-2*(ebyte+line_start): -2*line_start]
                        
                        line_start += ebyte
                        num = int( str, 16 )
                        result.append( num )                    

                else:
                    # handle mask register
                    # every hex char have 4 bits
                    for no in range(2*line_start, 64):
                        str = line[ 63-no ]
                        num = int(str, 16)
                        result.append( num >> 0 & 1 )
                        result.append( num >> 1 & 1 )
                        result.append( num >> 2 & 1 )
                        result.append( num >> 3 & 1 )
                
                now += 32
                if len(result) >= size:
                    # if we get enough result, break the loop
                    break

    # get size of element in result as final result
    result = result[:size]
    
    # update start
    if ebyte == 0.1:
        ebyte = 1
        # start's unit is byte and bits save in byte, so start plus enough bytes
        if size % 8 != 0:
            start += size // 8 + 1
        else:
            start += size / 8
    else:
        # start plus size of the ebyte
        start += size * ebyte 
         
    # make data into a np.ndarray and same dtype and shape with golden
    data = np.array(result, dtype='uint%d' % (ebyte*8))
    data.dtype = dtype
    data = data.reshape( golden.shape )

    return data

def diff_to_txt(a, b, filename, a_name, b_name):
    a = a.reshape(-1)
    b = b.reshape(-1)
    ah = a.copy()
    ah.dtype = f'uint{a.itemsize * 8}'
    bh = b.copy()
    bh.dtype = f'uint{b.itemsize * 8}'

    w = a.itemsize * 2
    if a.dtype == np.float16 or a.dtype == np.float32 or a.dtype == np.float64 or a.dtype == jnp.bfloat16:
        t = 'f'
    else:
        t = 'd'

    diff_result = True
    with open(filename, 'w') as file:
        print( f'         %{2*w+12}s  %{2*w+12}s' % ( a_name, b_name ), file=file)
        for i in range(a.shape[0]):
            if a[i] == b[i] or (np.isnan(a[i]) and np.isnan(b[i])):# np.array_equal( a[i], b[i], equal_nan=True)
                print(f'%8d: %{w+10}{t}(%0{w}x), %{w+10}{t}(%0{w}x)' % (i, a[i], ah[i], b[i], bh[i]), file=file)
            else:
                print(f'%8d: %{w+10}{t}(%0{w}x), %{w+10}{t}(%0{w}x), mismatch' % (i, a[i], ah[i], b[i], bh[i]), file=file)
                diff_result = False

    return diff_result

def check_to_txt(golden, result, filename, check_str):
    a = golden.reshape(-1)
    b = result.reshape(-1)
    ah = a.copy()
    ah.dtype = f'uint{a.itemsize * 8}'
    bh = b.copy()
    bh.dtype = f'uint{b.itemsize * 8}'

    w = a.itemsize * 2
    if a.dtype == np.float16 or a.dtype == np.float32 or a.dtype == np.float64 or a.dtype == jnp.bfloat16:
        t = 'f'
    else:
        t = 'd'

    check_result = True
    with open(filename, 'w') as file:
        print( f'         %{2*w+12}s  %{2*w+12}s' % ( 'golden', 'result' ), file=file)
        for i in range(a.shape[0]):
            golden = a[i]
            result = b[i]
            if eval(check_str):
                print(f'%8d: %{w+10}{t}(%0{w}x), %{w+10}{t}(%0{w}x)' % (i, a[i], ah[i], b[i], bh[i]), file=file)
            else:
                print(f'%8d: %{w+10}{t}(%0{w}x), %{w+10}{t}(%0{w}x), mismatch' % (i, a[i], ah[i], b[i], bh[i]), file=file)
                check_result = False
    
    return check_result


def get_sig_addr(file, sig):
    with open(file, 'r') as f:
        for line in f :
            if sig in line:
                addr = int(line.split()[1], 16)
                break
    return addr
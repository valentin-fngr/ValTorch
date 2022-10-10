import numpy as np 


def compute_dilation_shape(h, w, dilation): 
    """
        Computes the shapes of an 2D+ tensor with height and width (h, w)

        Args: 
            h (int) : the height 
            w (int) : the width 
            dilation (int) : dilation factor
        
        Output: 
            (tuple) : (new_height, new_width)
    """
    return (
        h + h*(dilation - 1) - (dilation - 1), 
        w + w*(dilation - 1) - (dilation - 1)
    )


def dilate_kernel(kernel, dilation): 

    """"
        Computes the dilated version of a kernel 

        Args: 
            kernel (tensor) : the input kernel 
                shape : (c_out, c_in, k1, k2)
            dilation (int) : dilation factor

        output: 
            new_kernel (tensor) : the output dilated kernel
                shape : (
                    c_in, 
                    c_out, 
                    k1 + k1*(dilation - 1) - (dilation - 1), 
                    k2 + k2*(dilation - 1) - (dilation - 1)
                )
    """


    row_idx, col_idx = 0, 0
    c_out, c_in, old_k1, old_k2 = kernel.shape
    k1, k2 = compute_dilation_shape(old_k1, old_k2, dilation)
    new_kernel = np.zeros((c_out, c_in, k1, k2))

    for i in range(k1): 
        for j in range(k2): 
            if i % dilation == 0 and j % dilation == 0: 
                if col_idx >= old_k1:
                    # go to next row
                    col_idx = 0 
                    row_idx += 1 
                
                if row_idx >= old_k2:
                    # no more values in the initial kernel
                    break
                
                new_kernel[:, :, i,j] = kernel[:,:,row_idx,col_idx]
                col_idx += 1 

            else: 
                new_kernel[:, :, i,j] = 0
    
        if row_idx >= old_k2: 
            break
        

    return new_kernel

        

def conv2d(input, kernel, bias=None,  stride=1, padding=0, dilation=1):
    """
        Args: 
            input (tensor) : input tensor
                shape : (N, c_in, h_in, w_in)
            kernel (tensor) : convolution kernel 
                shape : (c_out, c_in, k1, k2)
            bias : 
            stride (int) : convolution stride
            padding (int) : convolution padding 
            dilation (int) : convolution dilation

        Output: 
            output (tensor) : output tensor of the convolution 
                shape : (
                    N, 
                    c_out, 
                    (h_in + 2*padding - dilation * (k1 - 1) - 1) / stride + 1, 
                    (w_in + 2*padding - dilation * (k2 - 1) - 1) / stride + 1)
                )
    """
    
    assert input.ndim == 4
    assert kernel.ndim == 4 

    if bias is not None: 
        assert bias.ndim == 1 and bias.shape[0] == kernel.shape[0]

    assert stride >= 1 and dilation >= 1 and padding >= 0 


    if padding != 0: 
        # TODO : add other padding strategies 
        # pad input
        input = np.pad(input, padding)


    batch_size, _, h,w = input.shape[2:]
    c_out, c_in, k1, k2 = kernel.shape

    has_dilation = True if dilation > 1 else False 

    if has_dilation: 
        # apply dilation on the kernel
        kernel = dilate_kernel(kernel, dilation)
        _, _, k1, k2
        
    output_shape = (
        batch_size, 
        c_out,
        (h +2*padding - dilation * (k1 - 1*int(has_dilation)) - 1*int(has_dilation)) / stride + 1, 
        (w +2*padding - dilation * (k2 - 1*int(has_dilation)) - 1*int(has_dilation)) / stride + 1
    )


    output = np.zeros(output_shape)

    for i in range(0, h - k1, stride): 
        for j in range(0, w - k2, stride):
            
            output[:,:,i,j] = np.sum(input[:,:,i:i+k1, j:j+k2] * kernel[None, ...]) # ADD AXIS !!!!

    
    # readapt bias's shape
    if bias is not None: 
        return output + bias[None, :, None, None]
    return output

                 

    
     

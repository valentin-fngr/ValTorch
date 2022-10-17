import numpy as np 
import nn_from_scratch.nn.functional as F

class Layer: 

    """
        A basic module class that will be overwritten by each layers with learnable parameters
    """

    def __init__(self, weights, bias): 
        self.weights = weights 
        self.bias = bias 

    
    def forward(self, input): 
        pass

    def backward_delta(self): 
        pass 

    def backward(self): 
        pass 

    def update_parameters(self, weights, bias): 
        pass 



class Conv2d(Layer): 
    def __init__(self, weights, bias, stride=1, padding=0, dilation=1): 
        super().__init__(weights, bias)
        self.stride = stride 
        self.padding = padding 
        self.dilation = dilation 
        
    def forward(self, input):
        """
            Conv2d forward pass
        """

        output = F.conv2d(
            input, 
            self.weights, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation
        )

        return output 


    def backward(self, delta, input): 
        """
            Backward pass to compute loss gradient with respect to the layer's parameters
        """

        if self.bias is not None: 
            raise ValueError("We cannot compute gradient with respect to the bias yet !")
        
        dilation = self.stride
        gradient_weights = F.conv2d(input, delta, style="backward", dilation=dilation)
        return gradient_weights

    def backward_delta(self, previous_delta):
        """
            Backward pass to compute loss gradient with respect to the input of the layer

            Args: 
                previous_delta (4d tensor) : gradient with respect the input of the next layer (computed before in the backward pass)
            
            Output: 
                gradient_delta (4d tensor) 
                
        """

        if self.bias is not None: 
            raise ValueError("We cannot compute gradient with respect to the bias yet !")


        if self.stride > 1: 
            previous_delta = np.pad(F.dilate_array(previous_delta, self.stride), ((0,0), (0, 0), (self.stride, self.stride), (self.stride, self.stride)))
            padding = 0
            weights = np.flip(self.weights, axis=(2,3))
            gradient_delta = F.conv2d(
                previous_delta, 
                weights, 
                style="backward_delta",
                padding=padding
            )
        
        else: 
            weights = np.flip(self.weights, axis=(2, 3))
            padding = previous_delta.shape[3] - 1
            
            gradient_delta = F.conv2d(
                weights, 
                previous_delta, 
                style="backward_delta",
                padding=padding
            )

            gradient_delta = np.flip(gradient_delta, axis=(3, 2))

        return gradient_delta


        

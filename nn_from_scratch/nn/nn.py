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
            Backward pass to compute loss gradient with respect to the layer's parametres
        """

        if self.bias is not None: 
            raise ValueError("We cannot compute gradient with respect to the bias yet !")
        

        backward_weight = F.conv2d(input, delta, style="backward")
        return backward_weight
        

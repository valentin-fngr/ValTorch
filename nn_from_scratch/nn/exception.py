
class FloatingShapesError(ValueError): 
    """
        Raise when the convolution layer's hyper parametres
        produce an output shape with decimals
    """

    def __init__(self, message, *args, **kwargs): 
        self.stride = kwargs.get("stride")
        self.padding = kwargs.get("padding")
        self.kernel_size = kwargs.get("kernel_size")
        self.input_size = kwargs.get("input_size")
        self.message = message
        super().__init__(message, *args, **kwargs)
        
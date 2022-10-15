import unittest
from ..nn import nn as nn
import numpy as np 
import torch 
from ..nn.exception import * 


class TestConv2dLayer(unittest.TestCase): 

    def test_forward_pass_throws_wrong_parameters_shape(self): 

        w = 10
        k = 3
        c_out = 128
        stride = 2

        x = np.random.randn(10, 3, w, w).astype(np.float32)

        weights = np.random.randn(c_out, 3, k, k).astype(np.float32)
        bias = np.random.randn(c_out)

        conv_layer = nn.Conv2d(
            weights, 
            bias, 
            stride=stride
        )
        self.assertRaises(FloatingShapesError, conv_layer.forward, x)


    def test_forward_pass_dilation(self): 

        dilation = 2
        w = 5
        k = 2
        c_out = 128

        x = np.random.randn(10, 3, w, w).astype(np.float32)
        _x = torch.Tensor(x).detach()

        weights = np.random.randn(c_out, 3, k, k).astype(np.float32)
        _weights = torch.Tensor(weights).detach()

        bias = np.random.randn(c_out)
        _bias = torch.Tensor(bias).detach()

        conv_layer = nn.Conv2d(
            weights, 
            bias, 
            dilation=dilation
        )

        oracle_conv_layer = torch.nn.Conv2d(3, c_out, k, dilation=dilation)
        oracle_conv_layer.weight.data = _weights
        oracle_conv_layer.bias.data = _bias 

        output = conv_layer.forward(x)
        oracle_output = oracle_conv_layer(_x).detach()


        self.assertTrue(output.shape, oracle_output.shape)
        np.testing.assert_almost_equal(weights, _weights, decimal=4)
        np.testing.assert_almost_equal(output, oracle_output, decimal=4)


    def test_forward_pass_stride(self): 

        stride=2
        n = 10 
        c_in = 3
        w = 13
        k = 3
        c_out = 128

        x = np.random.randn(n, c_in, w, w).astype(np.float32)
        _x = torch.Tensor(x).detach()

        weights = np.random.randn(c_out, c_in, k, k).astype(np.float32)
        _weights = torch.Tensor(weights).detach()

        bias = np.random.randn(c_out)
        _bias = torch.Tensor(bias).detach()

        conv_layer = nn.Conv2d(
            weights, 
            bias, 
            stride=stride
        )

        oracle_conv_layer = torch.nn.Conv2d(c_in, c_out, k, stride=stride)
        oracle_conv_layer.weight.data = _weights
        oracle_conv_layer.bias.data = _bias 

        output = conv_layer.forward(x)
        oracle_output = oracle_conv_layer(_x).detach()

        self.assertTrue(output.shape, oracle_output.shape)
        np.testing.assert_almost_equal(weights, _weights, decimal=4)
        np.testing.assert_almost_equal(output, oracle_output, decimal=5)


    def test_backward_parameters(self): 
        
        
        n = 10 
        c_in = 3 
        w = 13
        c_out = 128
        k = 3

        x = np.random.randn(n, c_in, w, w)
        _x = torch.tensor(x, requires_grad=False).to(torch.double)

        weights = np.random.randn(c_out, c_in, k, k)
        _weights = torch.tensor(weights).to(torch.double)
        _weights.requires_grad = True
        
        bias = None

        conv_layer = nn.Conv2d(weights, bias)
        
        oracle_conv_layer = torch.nn.Conv2d(c_in, c_out, k)
        oracle_conv_layer.weight.data = _weights

        output = conv_layer.forward(x)
        oracle_output = oracle_conv_layer(_x)

        # backpropagating using auto grad 
        oracle_output.sum().backward()
        oracle_weights_grad = oracle_conv_layer.weight.grad

        # backpropagating using our framework 
        delta_grad = np.ones_like(output)
        weights_grad = conv_layer.backward(delta_grad, x)


        self.assertEqual(weights_grad.shape, weights.shape)
        np.testing.assert_almost_equal(weights_grad, oracle_weights_grad)




    def test_backward_parameters_with_stride(self): 
        
        stride=2
        n = 10 
        c_in = 3
        w = 13
        k = 3
        c_out = 128


        x = np.random.randn(n, c_in, w, w).astype(np.float32)
        _x = torch.Tensor(x).detach()

        weights = np.random.randn(c_out, c_in, k, k).astype(np.float32)
        _weights = torch.Tensor(weights).detach()

        bias = None

        conv_layer = nn.Conv2d(weights, bias=bias, stride=stride)
        
        oracle_conv_layer = torch.nn.Conv2d(c_in, c_out, k, stride=stride, bias=False)
        oracle_conv_layer.weight.data = _weights

        output = conv_layer.forward(x)
        oracle_output = oracle_conv_layer(_x)


        # backpropagating using auto grad 
        oracle_output.sum().backward()
        oracle_weights_grad = oracle_conv_layer.weight.grad


        # backpropagating using our framework 
        delta_grad = np.ones_like(output)
        weights_grad = conv_layer.backward(delta_grad, x)

        self.assertEqual(weights_grad.shape, weights.shape)
        np.testing.assert_almost_equal(weights_grad, oracle_weights_grad, decimal=4)


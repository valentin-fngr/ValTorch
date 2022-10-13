import unittest
from ..nn import nn as nn
import numpy as np 
import torch 


class TestConv2dLayer(unittest.TestCase): 

    def test_forward_pass_no_bias(self): 

        x = np.random.randn(10, 3, 3, 3).astype(np.float32)
        _x = torch.Tensor(x).detach()

        weights = np.random.randn(128, 3, 2, 2).astype(np.float32)
        _weights = torch.Tensor(weights).detach()

        bias = None

        conv_layer = nn.Conv2d(
            weights, 
            bias, 
            stride=1, 
            padding=0, 
            dilation=1
        )

        oracle_conv_layer = torch.nn.Conv2d(3, 10, 2, bias=False)
        oracle_conv_layer.weight.data = _weights
        
        output = conv_layer.forward(x)
        oracle_output = oracle_conv_layer(_x).detach()

        self.assertTrue(output.shape, oracle_output.shape)
        np.testing.assert_almost_equal(weights, _weights, decimal=4)
        np.testing.assert_almost_equal(output, oracle_output, decimal=4)

    def test_forward_pass_bias(self): 

        x = np.random.randn(10, 3, 3, 3).astype(np.float32)
        _x = torch.Tensor(x).detach()

        weights = np.random.randn(128, 3, 2, 2).astype(np.float32)
        _weights = torch.Tensor(weights).detach()

        bias = np.random.randn(128)
        _bias = torch.Tensor(bias).detach()

        conv_layer = nn.Conv2d(
            weights, 
            bias, 
            stride=1, 
            padding=0, 
            dilation=1
        )

        oracle_conv_layer = torch.nn.Conv2d(3, 10, 2)
        oracle_conv_layer.bias.data = _bias
        oracle_conv_layer.weight.data = _weights
        
        output = conv_layer.forward(x)
        oracle_output = oracle_conv_layer(_x).detach()

        self.assertTrue(output.shape, oracle_output.shape)
        np.testing.assert_almost_equal(weights, _weights, decimal=4)
        np.testing.assert_almost_equal(output, oracle_output, decimal=4)

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
        w = 5
        k = 3
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
            stride=stride
        )

        oracle_conv_layer = torch.nn.Conv2d(3, c_out, k, stride=stride)
        oracle_conv_layer.weight.data = _weights
        oracle_conv_layer.bias.data = _bias 

        output = conv_layer.forward(x)
        oracle_output = oracle_conv_layer(_x).detach()

        self.assertTrue(output.shape, oracle_output.shape)
        np.testing.assert_almost_equal(weights, _weights, decimal=4)
        np.testing.assert_almost_equal(output, oracle_output, decimal=4)


    def test_backward_parameters(self): 
        
        
        n = 10 
        c_in = 3 
        h, w = 10, 10
        c_out = 128
        k = 3

        x = np.random.randn(n, c_in, h, w)
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

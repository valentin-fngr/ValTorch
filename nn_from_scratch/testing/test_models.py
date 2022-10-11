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
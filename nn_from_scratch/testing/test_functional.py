import unittest
import numpy as np 
from ..nn import functional as F
from nn_from_scratch.nn.functional import dilate_kernel


class TestDilationFunctions(unittest.TestCase): 
    
    def test_can_dilate_kernel(self): 
        dilation = 2
        kernel = np.ones((1, 1, 3, 3))
        new_kernel = F.dilate_kernel(kernel, dilation)

        oracle_new_kernel = np.array([
            [1, 0, 1, 0, 1], 
            [0, 0, 0, 0, 0], 
            [1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0], 
            [1, 0, 1, 0, 1]
        ])

        expected_shape = F.compute_dilation_shape(3, 3, 2)

        self.assertEqual(new_kernel.shape[2:], expected_shape)
        assert np.array_equal(oracle_new_kernel, new_kernel[0, 0])


        kernel = np.ones((1, 1, 2, 2))
        new_kernel = F.dilate_kernel(kernel, dilation)
        expected_shape = F.compute_dilation_shape(2, 2, 2)

        oracle_new_kernel = np.array([
            [1, 0, 1], 
            [0, 0, 0], 
            [1, 0, 1]
        ])

        self.assertEqual(new_kernel.shape[2:], expected_shape)
        assert np.array_equal(oracle_new_kernel, new_kernel[0, 0])








if __name__ == '__main__':
    unittest.main()

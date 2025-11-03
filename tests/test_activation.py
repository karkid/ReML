import numpy as np
import pytest

from reml.activation import (
    sigmoid, dsigmoid,
    tanh, dtanh,
    relu, drelu,
    leaky_relu, dleaky_relu,
    softplus, dsoftplus,
    softmax, dsoftmax,
    elu, delu
)


class TestSigmoidActivation:
    """Test sigmoid activation function and its derivative."""
    
    def test_sigmoid_basic(self):
        """Test basic sigmoid functionality."""
        x = np.array([0, 1, -1, 2, -2])
        result = sigmoid(x)
        
        # Sigmoid should be between 0 and 1
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        
        # sigmoid(0) should be 0.5
        assert np.isclose(sigmoid(0), 0.5)
    
    def test_sigmoid_derivative(self):
        """Test sigmoid derivative."""
        x = np.array([0, 1, -1])
        dy = dsigmoid(x)
        
        # Derivative should be positive
        assert np.all(dy >= 0)
        
        # dsigmoid(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        assert np.isclose(dsigmoid(0), 0.25)
    
    def test_sigmoid_numerical_stability(self):
        """Test sigmoid with large values."""
        large_pos = np.array([100, 500, 1000])
        large_neg = np.array([-100, -500, -1000])
        
        # Should not overflow or underflow
        pos_result = sigmoid(large_pos)
        neg_result = sigmoid(large_neg)
        
        assert np.all(np.isfinite(pos_result))
        assert np.all(np.isfinite(neg_result))
        assert np.allclose(pos_result, 1.0, atol=1e-10)
        assert np.allclose(neg_result, 0.0, atol=1e-10)


class TestTanhActivation:
    """Test tanh activation function and its derivative."""
    
    def test_tanh_basic(self):
        """Test basic tanh functionality."""
        x = np.array([0, 1, -1, 2, -2])
        result = tanh(x)
        
        # Tanh should be between -1 and 1
        assert np.all(result >= -1)
        assert np.all(result <= 1)
        
        # tanh(0) should be 0
        assert np.isclose(tanh(0), 0)
    
    def test_tanh_derivative(self):
        """Test tanh derivative."""
        x = np.array([0, 1, -1])
        dy = dtanh(x)
        
        # Derivative should be positive
        assert np.all(dy >= 0)
        
        # dtanh(0) = 1 - tanh(0)^2 = 1 - 0 = 1
        assert np.isclose(dtanh(0), 1.0)
    
    def test_tanh_symmetry(self):
        """Test tanh antisymmetry property."""
        x = np.array([1, 2, 3])
        assert np.allclose(tanh(-x), -tanh(x))


class TestReLUActivation:
    """Test ReLU activation function and its derivative."""
    
    def test_relu_basic(self):
        """Test basic ReLU functionality."""
        x = np.array([-2, -1, 0, 1, 2])
        result = relu(x)
        
        expected = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_equal(result, expected)
    
    def test_relu_derivative(self):
        """Test ReLU derivative."""
        x = np.array([-2, -1, 0, 1, 2])
        dy = drelu(x)
        
        expected = np.array([0, 0, 0, 1, 1])
        np.testing.assert_array_equal(dy, expected)
    
    def test_relu_non_negativity(self):
        """Test ReLU non-negativity property."""
        x = np.random.randn(100)
        result = relu(x)
        assert np.all(result >= 0)


class TestLeakyReLUActivation:
    """Test Leaky ReLU activation function and its derivative."""
    
    def test_leaky_relu_basic(self):
        """Test basic Leaky ReLU functionality."""
        x = np.array([-2, -1, 0, 1, 2])
        result = leaky_relu(x, alpha=0.1)
        
        expected = np.array([-0.2, -0.1, 0, 1, 2])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_leaky_relu_derivative(self):
        """Test Leaky ReLU derivative."""
        x = np.array([-2, -1, 0, 1, 2])
        dy = dleaky_relu(x, alpha=0.1)
        
        expected = np.array([0.1, 0.1, 1.0, 1, 1])  # x=0 gets derivative 1.0, not alpha
        np.testing.assert_array_almost_equal(dy, expected)
    
    def test_leaky_relu_alpha_parameter(self):
        """Test Leaky ReLU with different alpha values."""
        x = np.array([-1])
        
        result_01 = leaky_relu(x, alpha=0.1)
        result_02 = leaky_relu(x, alpha=0.2)
        
        assert result_01[0] == -0.1
        assert result_02[0] == -0.2


class TestSoftplusActivation:
    """Test Softplus activation function and its derivative."""
    
    def test_softplus_basic(self):
        """Test basic Softplus functionality."""
        x = np.array([0, 1, -1])
        result = softplus(x)
        
        # Softplus should be positive
        assert np.all(result > 0)
        
        # softplus(0) ≈ ln(2)
        assert np.isclose(softplus(0), np.log(2))
    
    def test_softplus_derivative(self):
        """Test Softplus derivative is sigmoid."""
        x = np.array([0, 1, -1, 2])
        dy_softplus = dsoftplus(x)
        sigmoid_result = sigmoid(x)
        
        np.testing.assert_array_almost_equal(dy_softplus, sigmoid_result)
    
    def test_softplus_approximates_relu(self):
        """Test that Softplus approximates ReLU for large positive values."""
        x = np.array([10, 20, 50])
        result = softplus(x)
        
        # For large x, softplus(x) ≈ x
        np.testing.assert_allclose(result, x, atol=1e-4)  # More reasonable tolerance


class TestSoftmaxActivation:
    """Test Softmax activation function and its derivative."""
    
    def test_softmax_basic(self):
        """Test basic Softmax functionality."""
        x = np.array([1, 2, 3])
        result = softmax(x)
        
        # Should sum to 1
        assert np.isclose(np.sum(result), 1.0)
        
        # All values should be positive
        assert np.all(result > 0)
    
    def test_softmax_numerical_stability(self):
        """Test Softmax numerical stability with large values."""
        x = np.array([1000, 1001, 1002])
        result = softmax(x)
        
        # Should not overflow
        assert np.all(np.isfinite(result))
        assert np.isclose(np.sum(result), 1.0)
    
    def test_softmax_derivative_shape(self):
        """Test Softmax derivative shape."""
        x = np.array([1, 2, 3])
        jacobian = dsoftmax(x)
        
        # Should be n x n matrix
        assert jacobian.shape == (len(x), len(x))
    
    def test_softmax_invariance(self):
        """Test Softmax translation invariance."""
        x = np.array([1, 2, 3])
        c = 10
        
        result1 = softmax(x)
        result2 = softmax(x + c)
        
        np.testing.assert_allclose(result1, result2)


class TestELUActivation:
    """Test ELU activation function and its derivative."""
    
    def test_elu_basic(self):
        """Test basic ELU functionality."""
        x = np.array([-2, -1, 0, 1, 2])
        result = elu(x, alpha=1.0)
        
        # For x >= 0, ELU(x) = x
        positive_mask = x >= 0
        np.testing.assert_array_equal(result[positive_mask], x[positive_mask])
        
        # For x < 0, ELU(x) = alpha * (exp(x) - 1)
        negative_mask = x < 0
        expected_negative = 1.0 * (np.exp(x[negative_mask]) - 1)
        np.testing.assert_array_almost_equal(result[negative_mask], expected_negative)
    
    def test_elu_derivative(self):
        """Test ELU derivative."""
        x = np.array([-2, -1, 0, 1, 2])
        dy = delu(x, alpha=1.0)
        
        # For x > 0, derivative should be 1
        positive_mask = x > 0
        np.testing.assert_array_equal(dy[positive_mask], 1.0)
        
        # For x < 0, derivative should be alpha * exp(x)
        negative_mask = x < 0
        expected_negative = 1.0 * np.exp(x[negative_mask])
        np.testing.assert_array_almost_equal(dy[negative_mask], expected_negative)
    
    def test_elu_alpha_parameter(self):
        """Test ELU with different alpha values."""
        x = np.array([-1])
        
        result_1 = elu(x, alpha=1.0)
        result_2 = elu(x, alpha=2.0)
        
        # With alpha=2, negative values should be scaled by 2
        assert np.isclose(result_2[0], 2 * (np.exp(-1) - 1))


class TestActivationProperties:
    """Test general properties of activation functions."""
    
    def test_activation_derivative_consistency(self):
        """Test that derivatives are consistent with numerical gradients."""
        x = np.array([0.5])
        h = 1e-7
        
        # Test sigmoid
        numerical_grad = (sigmoid(x + h) - sigmoid(x - h)) / (2 * h)
        analytical_grad = dsigmoid(x)
        assert np.isclose(numerical_grad, analytical_grad, atol=1e-5)
        
        # Test tanh
        numerical_grad = (tanh(x + h) - tanh(x - h)) / (2 * h)
        analytical_grad = dtanh(x)
        assert np.isclose(numerical_grad, analytical_grad, atol=1e-5)
    
    def test_activation_function_ranges(self):
        """Test that activation functions produce expected output ranges."""
        x = np.linspace(-10, 10, 100)
        
        # Sigmoid: (0, 1)
        sigmoid_result = sigmoid(x)
        assert np.all(sigmoid_result > 0)
        assert np.all(sigmoid_result < 1)
        
        # Tanh: (-1, 1)
        tanh_result = tanh(x)
        assert np.all(tanh_result > -1)
        assert np.all(tanh_result < 1)
        
        # ReLU: [0, inf)
        relu_result = relu(x)
        assert np.all(relu_result >= 0)
        
        # Softplus: (0, inf)
        softplus_result = softplus(x)
        assert np.all(softplus_result > 0)


class TestActivationEdgeCases:
    """Test edge cases for activation functions."""
    
    def test_empty_input(self):
        """Test activation functions with empty input."""
        x = np.array([])
        
        assert sigmoid(x).shape == (0,)
        assert tanh(x).shape == (0,)
        assert relu(x).shape == (0,)
        assert softplus(x).shape == (0,)
    
    def test_single_value_input(self):
        """Test activation functions with single value input."""
        x = 2.0
        
        # Should work with scalar input
        assert np.isscalar(sigmoid(x))
        assert np.isscalar(tanh(x))
        assert np.isscalar(relu(x))
        assert np.isscalar(softplus(x))
    
    def test_zero_input(self):
        """Test activation functions with zero input."""
        x = np.array([0.0])
        
        assert np.isclose(sigmoid(x), 0.5)
        assert np.isclose(tanh(x), 0.0)
        assert np.isclose(relu(x), 0.0)
        assert np.isclose(softplus(x), np.log(2))
        assert np.isclose(elu(x), 0.0)
    
    def test_inf_input_handling(self):
        """Test activation functions with infinite input."""
        pos_inf = np.array([np.inf])
        neg_inf = np.array([-np.inf])
        
        # Sigmoid
        assert np.isclose(sigmoid(pos_inf), 1.0)
        assert np.isclose(sigmoid(neg_inf), 0.0)
        
        # Tanh
        assert np.isclose(tanh(pos_inf), 1.0)
        assert np.isclose(tanh(neg_inf), -1.0)
        
        # ReLU
        assert relu(pos_inf) == pos_inf
        assert relu(neg_inf) == 0.0
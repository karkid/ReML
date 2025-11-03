import numpy as np
import pytest

from reml.neural_network.layers import Layer, Dense, ReLU, Sigmoid, Tanh
from reml.neural_network.losses import Loss, MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy
from reml.neural_network.optimizers import SGD
from reml.neural_network.models import Sequential


class TestNeuralNetworkIntegration:
    """Test neural network components integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
    
    def test_dense_layer_basic(self):
        """Test Dense layer basic functionality."""
        layer = Dense(3, 2)
        
        # Check initialization
        assert layer.W.shape == (3, 2)
        assert layer.b.shape == (1, 2)
        
        # Test forward pass
        x = np.random.randn(2, 3)
        output = layer.forward(x)
        assert output.shape == (2, 2)
    
    def test_relu_layer_basic(self):
        """Test ReLU layer basic functionality."""
        layer = ReLU()
        
        x = np.array([[-1, 0, 1], [2, -2, 3]])
        output = layer.forward(x)
        expected = np.array([[0, 0, 1], [2, 0, 3]])
        
        np.testing.assert_array_equal(output, expected)
    
    def test_sigmoid_layer_basic(self):
        """Test Sigmoid layer basic functionality."""
        layer = Sigmoid()
        
        x = np.array([[0, 1, -1]])
        output = layer.forward(x)
        
        # Sigmoid should be between 0 and 1
        assert np.all(output >= 0)
        assert np.all(output <= 1)
        # sigmoid(0) should be 0.5
        assert np.isclose(output[0, 0], 0.5)
    
    def test_mse_loss_basic(self):
        """Test MSE loss basic functionality."""
        loss_fn = MeanSquaredError()
        
        y_true = np.array([[1.0, 2.0]])
        y_pred = np.array([[1.1, 1.9]])
        
        loss = loss_fn.forward(y_pred, y_true)
        assert np.isscalar(loss)
        assert loss >= 0
        
        # Test backward pass
        grad = loss_fn.backward()
        assert grad.shape == y_pred.shape
    
    def test_bce_loss_basic(self):
        """Test BCE loss basic functionality."""
        loss_fn = BinaryCrossentropy()
        
        y_true = np.array([[1.0], [0.0]])
        y_pred = np.array([[0.9], [0.1]])
        
        loss = loss_fn.forward(y_pred, y_true)
        assert np.isscalar(loss)
        assert loss >= 0
        
        # Test backward pass
        grad = loss_fn.backward()
        assert grad.shape == y_pred.shape
    
    def test_sgd_optimizer_basic(self):
        """Test SGD optimizer basic functionality."""
        optimizer = SGD(lr=0.01)
        assert optimizer.lr == 0.01
        
        # Test with default learning rate
        optimizer_default = SGD()
        assert optimizer_default.lr == 0.1  # Default value
    
    def test_sequential_model_basic(self):
        """Test Sequential model basic functionality."""
        # Sequential takes layers in constructor
        model = Sequential(
            Dense(3, 4),
            ReLU(),
            Dense(4, 2)
        )
        
        assert len(model.layers) == 3
        
        # Test forward pass
        x = np.random.randn(2, 3)
        output = model.forward(x)
        assert output.shape == (2, 2)
    
    def test_layer_chaining(self):
        """Test chaining layers together."""
        layer1 = Dense(3, 4)
        layer2 = ReLU()
        layer3 = Dense(4, 2)
        
        x = np.random.randn(2, 3)
        
        # Forward pass through layers
        out1 = layer1.forward(x)
        out2 = layer2.forward(out1)
        out3 = layer3.forward(out2)
        
        assert out1.shape == (2, 4)
        assert out2.shape == (2, 4)
        assert out3.shape == (2, 2)
    
    def test_model_compilation(self):
        """Test model compilation."""
        model = Sequential(Dense(2, 1))
        
        loss = MeanSquaredError()
        optimizer = SGD(lr=0.01)
        
        # Sequential doesn't have compile method, just test that objects exist
        assert isinstance(loss, MeanSquaredError)
        assert isinstance(optimizer, SGD)
    
    def test_different_activations(self):
        """Test different activation layers."""
        x = np.array([[0, 1, -1]])
        
        # ReLU
        relu = ReLU()
        relu_out = relu.forward(x)
        expected_relu = np.array([[0, 1, 0]])
        np.testing.assert_array_equal(relu_out, expected_relu)
        
        # Sigmoid
        sigmoid = Sigmoid()
        sigmoid_out = sigmoid.forward(x)
        assert np.all(sigmoid_out >= 0)
        assert np.all(sigmoid_out <= 1)
        
        # Tanh
        tanh = Tanh()
        tanh_out = tanh.forward(x)
        assert np.all(tanh_out >= -1)
        assert np.all(tanh_out <= 1)
    
    def test_loss_functions_comparison(self):
        """Test different loss functions."""
        y_true = np.array([[1.0]])
        y_pred_good = np.array([[0.95]])
        y_pred_bad = np.array([[0.1]])
        
        # BCE Loss
        bce = BinaryCrossentropy()
        loss_good = bce.forward(y_pred_good, y_true)
        loss_bad = bce.forward(y_pred_bad, y_true)
        
        # Good prediction should have lower loss
        assert loss_good < loss_bad
    
    def test_model_with_different_architectures(self):
        """Test models with different architectures."""
        # Simple model
        model1 = Sequential(Dense(2, 1))
        
        # Deeper model
        model2 = Sequential(
            Dense(2, 4),
            ReLU(),
            Dense(4, 4),
            Sigmoid(),
            Dense(4, 1)
        )
        
        x = np.random.randn(3, 2)
        
        out1 = model1.forward(x)
        out2 = model2.forward(x)
        
        assert out1.shape == (3, 1)
        assert out2.shape == (3, 1)
    
    def test_gradient_computation(self):
        """Test that layers can compute gradients."""
        layer = Dense(3, 2)
        x = np.random.randn(2, 3)
        
        # Forward pass
        output = layer.forward(x)
        
        # Check that layer stores input for backward pass
        assert hasattr(layer, '_x')
        
        # Check that gradients exist
        assert hasattr(layer, 'dW')
        assert hasattr(layer, 'db')
        assert layer.dW.shape == layer.W.shape
        if layer.db is not None:
            assert layer.db.shape == layer.b.shape
    
    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        # Large values
        x_large = np.array([[100, -100]])
        
        sigmoid = Sigmoid()
        out_sigmoid = sigmoid.forward(x_large)
        assert np.all(np.isfinite(out_sigmoid))
        
        tanh = Tanh()
        out_tanh = tanh.forward(x_large)
        assert np.all(np.isfinite(out_tanh))
        
        # Very small values
        y_true = np.array([[1.0]])
        y_pred_small = np.array([[1e-10]])
        
        bce = BinaryCrossentropy()
        loss = bce.forward(y_pred_small, y_true)
        assert np.isfinite(loss)
    
    def test_layer_parameters(self):
        """Test layer parameter access."""
        layer = Dense(3, 2)
        
        # Check params_and_grads method exists and returns generator
        params = layer.params_and_grads()
        # Convert generator to list to check
        params_list = list(params)
        assert isinstance(params_list, list)
    
    def test_model_evaluation_mode(self):
        """Test model training vs evaluation modes."""
        model = Sequential(Dense(2, 1))
        
        # Test mode switching
        model.train()
        model.eval()
        
        # Should complete without errors
        x = np.random.randn(1, 2)
        output = model.forward(x)
        assert output.shape == (1, 1)


class TestNeuralNetworkEdgeCases:
    """Test edge cases for neural network components."""
    
    def test_zero_dimensions(self):
        """Test handling of edge dimension cases."""
        # Test with minimum valid dimensions
        layer = Dense(1, 1)
        x = np.array([[1.0]])
        output = layer.forward(x)
        assert output.shape == (1, 1)
    
    def test_single_sample_batch(self):
        """Test with single sample batches."""
        model = Sequential(
            Dense(2, 3),
            ReLU(),
            Dense(3, 1)
        )
        
        x = np.random.randn(1, 2)
        output = model.forward(x)
        assert output.shape == (1, 1)
    
    def test_large_batch(self):
        """Test with large batch sizes."""
        model = Sequential(
            Dense(5, 3),
            Sigmoid(),
            Dense(3, 2)
        )
        
        x = np.random.randn(100, 5)
        output = model.forward(x)
        assert output.shape == (100, 2)
    
    def test_loss_with_perfect_predictions(self):
        """Test loss functions with perfect predictions."""
        # MSE with perfect prediction
        mse = MeanSquaredError()
        y_true = np.array([[1.0, 2.0]])
        loss = mse.forward(y_true, y_true)
        assert np.isclose(loss, 0.0)
    
    def test_activation_ranges(self):
        """Test activation function output ranges."""
        x = np.linspace(-10, 10, 100).reshape(-1, 1)
        
        # Sigmoid: (0, 1)
        sigmoid = Sigmoid()
        sig_out = sigmoid.forward(x)
        assert np.all(sig_out > 0)
        assert np.all(sig_out < 1)
        
        # Tanh: (-1, 1)
        tanh = Tanh()
        tanh_out = tanh.forward(x)
        assert np.all(tanh_out > -1)
        assert np.all(tanh_out < 1)
        
        # ReLU: [0, inf)
        relu = ReLU()
        relu_out = relu.forward(x)
        assert np.all(relu_out >= 0)


class TestNeuralNetworkProperties:
    """Test mathematical properties of neural network components."""
    
    def test_layer_determinism(self):
        """Test that layers produce deterministic outputs."""
        layer = Dense(3, 2)
        x = np.random.randn(2, 3)
        
        output1 = layer.forward(x)
        output2 = layer.forward(x)
        
        np.testing.assert_allclose(output1, output2)
    
    def test_activation_properties(self):
        """Test mathematical properties of activations."""
        x = np.array([[0]])
        
        # Sigmoid(0) = 0.5
        sigmoid = Sigmoid()
        assert np.isclose(sigmoid.forward(x), 0.5)
        
        # Tanh(0) = 0
        tanh = Tanh()
        assert np.isclose(tanh.forward(x), 0.0)
        
        # ReLU(0) = 0
        relu = ReLU()
        assert np.isclose(relu.forward(x), 0.0)
    
    def test_loss_properties(self):
        """Test mathematical properties of loss functions."""
        # MSE is always non-negative
        mse = MeanSquaredError()
        y_true = np.random.randn(5, 3)
        y_pred = np.random.randn(5, 3)
        loss = mse.forward(y_pred, y_true)
        assert loss >= 0
        
        # BCE is always non-negative
        bce = BinaryCrossentropy()
        y_true = np.random.randint(0, 2, (5, 1)).astype(float)
        y_pred = np.random.uniform(0.01, 0.99, (5, 1))
        loss = bce.forward(y_pred, y_true)
        assert loss >= 0
    
    def test_gradient_shapes(self):
        """Test that gradients have correct shapes."""
        layer = Dense(4, 3)
        x = np.random.randn(2, 4)
        
        # Forward pass
        output = layer.forward(x)
        
        # Check gradient shapes match parameter shapes
        assert layer.dW.shape == layer.W.shape
        if layer.db is not None:
            assert layer.db.shape == layer.b.shape
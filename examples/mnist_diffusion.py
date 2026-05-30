import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neutro.models.vision.diffusion_model import DiffusionModel
from neutro.models.vision.unet import UNet
from neutro.optimizers import Adam
from neutro.utils.data_utils import load_mnist

def train_diffusion():
    print("Loading MNIST...")
    (x_train, _), (x_test, _) = load_mnist()

    # Preprocess: scale to [-1, 1] for diffusion
    x_train = (x_train.reshape(-1, 28, 28, 1).astype('float32') / 127.5) - 1.0
    x_test = (x_test.reshape(-1, 28, 28, 1).astype('float32') / 127.5) - 1.0
    
    # Use smaller config for faster demo
    timesteps = 100
    base_filters = 16
    
    unet = UNet(input_channels=1, base_filters=base_filters)
    model = DiffusionModel(unet, timesteps=timesteps)
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    print("Starting training (Subset)...")
    # Small subset for demo
    x_sub = x_train[:500]
    
    epochs = 50
    batch_size = 32
    
    # Diffusion training: y_true is ignored in our forward pass, 
    # but Model.fit expects y. The model predicts noise and compares with sampled noise internally.
    # Wait, let's check DiffusionModel.forward again.
    # It returns predicted_noise.
    # So we should compare it with actual noise.
    
    # Actually, Model.fit does:
    # output = self.forward(x_batch, training=True)
    # batch_loss = self.loss_fn(y_batch, output)
    # grad = self.loss_fn.gradient(y_batch, output)
    # self.backward(grad)
    
    # DiffusionModel.forward returns predicted_noise.
    # It stores self.current_noise.
    # So the loss should be between self.current_noise and predicted_noise.
    # BUT Model.fit passes y_batch to loss_fn.
    
    # To make it work with Sequential.fit, we can:
    # 1. Pass actual noise as Y in fit.
    # 2. Modify DiffusionModel.forward to return predicted_noise and somehow handle noise.
    
    # Let's check DiffusionModel.forward again.
    # It generates noise internally.
    
    # I'll modify DiffusionModel.forward to return BOTH, or just return predicted_noise 
    # and we pass dummy Y, but then loss_fn will be wrong.
    
    # Better: return predicted_noise and let VAELoss-like custom loss handle it?
    # Or just return predicted_noise and pass the internal noise to the loss.
    
    print("Adjusting DiffusionModel for training...")
    
    class DiffusionLoss:
        def __init__(self, model):
            self.model = model
        def __call__(self, y_true, y_pred):
            return np.mean(np.square(self.model.current_noise - y_pred))
        def gradient(self, y_true, y_pred):
            return -2 * (self.model.current_noise - y_pred) / y_pred.size

    loss_fn = DiffusionLoss(model)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=loss_fn)
    
    model.fit(x_sub, x_sub, epochs=epochs, batch_size=batch_size)

    print("Sampling from Diffusion Model...")
    # Sampling is slow, let's do 1 sample
    sampled = model.sample(shape=(1, 28, 28, 1))
    
    # Save sampled image
    plt.imshow(sampled[0, :, :, 0], cmap='gray')
    plt.title("Generated Image (Diffusion)")
    os.makedirs('generated_diffusion', exist_ok=True)
    plt.savefig('generated_diffusion/sample.png')
    # plt.close()
    
    print(f"Sampled image shape: {sampled.shape}")

if __name__ == "__main__":
    train_diffusion()

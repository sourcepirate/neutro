import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neutro.models.vision.vae import VAE
from neutro.losses.vae_loss import VAELoss
from neutro.optimizers import Adam
from neutro.utils.data_utils import load_mnist
from neutro.preprocessing.image import ImageDataGenerator

def save_generated_images(model, epoch, latent_dim, n=10):
    """Saves a grid of generated images."""
    # Sample from latent space
    z = np.random.normal(size=(n * n, latent_dim))
    
    # We need to call the decoder part of the VAE
    # In our VAE implementation, forward(z) from the bottleneck onwards 
    # but the VAE.forward takes input x.
    # Let's use a trick: pass z to decoder_fc directly if possible, 
    # or just use model.decoder_fc and then reshape.
    
    reconstructions = model.decoder_fc.forward(z)
    reconstructions = reconstructions.reshape(-1, 28, 28)
    
    plt.figure(figsize=(10, 10))
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.imshow(reconstructions[i], cmap='gray')
        plt.axis('off')
    
    os.makedirs('generated_vae', exist_ok=True)
    plt.savefig(f'generated_vae/epoch_{epoch}.png')
    plt.close()

def train_vae():
    print("Loading MNIST...")
    (x_train, _), (x_test, _) = load_mnist()

    # Preprocess
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    latent_dim = 2
    input_shape = (28, 28, 1)
    
    model = VAE(input_shape=input_shape, latent_dim=latent_dim)
    
    # Custom training loop or model.compile + fit
    # VAELoss needs the model instance to access z_mean and z_log_var
    loss_fn = VAELoss(model, reconstruction_loss_type='bce', kl_weight=1.0)
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss=loss_fn)

    print("Starting training (Subset)...")
    # Small subset for demo
    x_sub = x_train[:2000]
    
    epochs = 50
    batch_size = 32
    
    # Since VAE is unsupervised, y_true is the same as x
    model.fit(x_sub, x_sub, epochs=epochs, batch_size=batch_size, validation_data=(x_test[:200], x_test[:200]))

    print("Generating images...")
    save_generated_images(model, "final", latent_dim)
    print("Results saved in generated_vae/")

if __name__ == "__main__":
    train_vae()

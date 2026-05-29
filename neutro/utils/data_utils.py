import os
import urllib.request
import gzip
import numpy as np
import ssl

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {url}...")
        # Bypass SSL verification for demo datasets if needed
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=context) as response, open(filename, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Downloaded {filename}")

def load_mnist():
    """Loads the MNIST dataset."""
    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    cache_dir = os.path.expanduser("~/.neutro/datasets")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "mnist.npz")
    
    download_file(base_url, path)
    
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_test'], f['y_test'] # Use test set as training for faster demo if needed, but let's use full
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        
    return (x_train, y_train), (x_test, y_test)

def load_wikitext2():
    """Loads the WikiText-2 dataset."""
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    cache_dir = os.path.expanduser("~/.neutro/datasets")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "wikitext2_train.txt")
    
    download_file(url, path)
    
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

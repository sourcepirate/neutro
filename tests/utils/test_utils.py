import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from neutro.utils.visualization import plot_attention_weights, print_text_with_probs
from neutro.utils.data_utils import download_file, load_mnist, load_wikitext2

def test_plot_attention_weights():
    # Smoke test to ensure it doesn't crash
    weights = np.random.rand(5, 5)
    plot_attention_weights(weights, tokens=["a", "b", "c", "d", "e"])
    plot_attention_weights(weights) # test default tokens

def test_print_text_with_probs():
    # Smoke test to ensure it doesn't crash
    tokens = ["hello", "world"]
    probs = [0.05, 0.95]
    print_text_with_probs(tokens, probs)

@patch('urllib.request.urlopen')
@patch('os.path.exists')
@patch('builtins.open', new_callable=MagicMock)
def test_download_file(mock_open, mock_exists, mock_urlopen):
    mock_exists.return_value = False
    mock_response = MagicMock()
    mock_response.read.return_value = b"dummy content"
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response
    
    download_file("http://example.com", "dummy.txt")
    
    mock_urlopen.assert_called_once()
    mock_open.assert_called_with("dummy.txt", "wb")

@patch('neutro.utils.data_utils.download_file')
@patch('numpy.load')
@patch('os.path.expanduser')
def test_load_mnist(mock_expanduser, mock_load, mock_download):
    mock_expanduser.return_value = "/tmp"
    
    mock_data = MagicMock()
    mock_data.__enter__.return_value = {
        'x_train': np.zeros((10, 28, 28)),
        'y_train': np.zeros(10),
        'x_test': np.zeros((2, 28, 28)),
        'y_test': np.zeros(2)
    }
    mock_load.return_value = mock_data
    
    (x_train, y_train), (x_test, y_test) = load_mnist()
    assert x_train.shape == (10, 28, 28)

@patch('neutro.utils.data_utils.download_file')
@patch('builtins.open', new_callable=MagicMock)
@patch('os.path.expanduser')
def test_load_wikitext2(mock_expanduser, mock_open, mock_download):
    mock_expanduser.return_value = "/tmp"
    mock_file = MagicMock()
    mock_file.read.return_value = "dummy text"
    mock_file.__enter__.return_value = mock_file
    mock_open.return_value = mock_file
    
    text = load_wikitext2()
    assert text == "dummy text"

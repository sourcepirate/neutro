import numpy as np
import pytest
import json
from unittest.mock import patch, MagicMock
from neutro.utils.data_utils import load_imdb, get_imdb_word_index


@patch('neutro.utils.data_utils.download_file')
@patch('numpy.load')
@patch('os.path.expanduser')
def test_load_imdb(mock_expanduser, mock_load, mock_download):
    mock_expanduser.return_value = "/tmp"
    mock_data = MagicMock()
    mock_data.__enter__.return_value = {
        'x_train': np.zeros((100,)),
        'y_train': np.zeros(100),
        'x_test': np.zeros((20,)),
        'y_test': np.zeros(20)
    }
    mock_load.return_value = mock_data
    
    (x_train, y_train), (x_test, y_test) = load_imdb()
    assert x_train.shape == (100,)
    assert y_train.shape == (100,)
    assert x_test.shape == (20,)
    mock_download.assert_called_once()


@patch('neutro.utils.data_utils.download_file')
@patch('builtins.open', new_callable=MagicMock)
@patch('json.load')
@patch('os.path.expanduser')
def test_get_imdb_word_index(mock_expanduser, mock_json_load, mock_open, mock_download):
    mock_expanduser.return_value = "/tmp"
    mock_json_load.return_value = {"the": 1, "and": 2, "a": 3}
    mock_file = MagicMock()
    mock_file.__enter__.return_value = mock_file
    mock_open.return_value = mock_file
    
    word_index = get_imdb_word_index()
    assert word_index == {"the": 1, "and": 2, "a": 3}
    mock_download.assert_called_once()

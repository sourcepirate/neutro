import numpy as np
import pytest
from neutro.preprocessing.sequence import pad_sequences


class TestPadSequencesBasic:
    def test_padding_pre_default(self):
        sequences = [[1, 2], [3, 4, 5]]
        result = pad_sequences(sequences, maxlen=3)
        expected = np.array([[0, 1, 2], [3, 4, 5]], dtype="int32")
        np.testing.assert_array_equal(result, expected)

    def test_padding_post(self):
        sequences = [[1, 2], [3, 4, 5]]
        result = pad_sequences(sequences, maxlen=3, padding="post")
        expected = np.array([[1, 2, 0], [3, 4, 5]], dtype="int32")
        np.testing.assert_array_equal(result, expected)


class TestPadSequencesTruncating:
    def test_truncating_pre(self):
        sequences = [[1, 2, 3, 4, 5]]
        result = pad_sequences(sequences, maxlen=3, truncating="pre")
        expected = np.array([[3, 4, 5]], dtype="int32")
        np.testing.assert_array_equal(result, expected)

    def test_truncating_post(self):
        sequences = [[1, 2, 3, 4, 5]]
        result = pad_sequences(sequences, maxlen=3, truncating="post")
        expected = np.array([[1, 2, 3]], dtype="int32")
        np.testing.assert_array_equal(result, expected)


class TestPadSequencesMaxlen:
    def test_custom_maxlen_shorter_than_longest(self):
        sequences = [[1, 2, 3, 4, 5], [1, 2]]
        result = pad_sequences(sequences, maxlen=3)
        expected = np.array([[3, 4, 5], [0, 1, 2]], dtype="int32")
        np.testing.assert_array_equal(result, expected)

    def test_maxlen_none_auto_detect(self):
        sequences = [[1, 2], [3, 4, 5, 6], [7]]
        result = pad_sequences(sequences, maxlen=None)
        expected = np.array(
            [[0, 0, 1, 2], [3, 4, 5, 6], [0, 0, 0, 7]], dtype="int32"
        )
        np.testing.assert_array_equal(result, expected)


class TestPadSequencesDtypeAndValue:
    def test_custom_dtype(self):
        sequences = [[1, 2], [3, 4, 5]]
        result = pad_sequences(sequences, maxlen=3, dtype="float32")
        assert result.dtype == np.float32
        expected = np.array([[0, 1, 2], [3, 4, 5]], dtype="float32")
        np.testing.assert_array_equal(result, expected)

    def test_custom_padding_value(self):
        sequences = [[1, 2], [3, 4, 5]]
        result = pad_sequences(sequences, maxlen=3, value=99)
        expected = np.array([[99, 1, 2], [3, 4, 5]], dtype="int32")
        np.testing.assert_array_equal(result, expected)


class TestPadSequencesEdgeCases:
    def test_empty_sequence_in_list(self):
        sequences = [[1, 2, 3], [], [4, 5]]
        result = pad_sequences(sequences, maxlen=3)
        expected = np.array([[1, 2, 3], [0, 0, 0], [0, 4, 5]], dtype="int32")
        np.testing.assert_array_equal(result, expected)


class TestPadSequencesErrors:
    def test_invalid_truncating_type(self):
        sequences = [[1, 2, 3]]
        with pytest.raises(ValueError, match='Truncating type "middle" not understood'):
            pad_sequences(sequences, maxlen=2, truncating="middle")

    def test_invalid_padding_type(self):
        sequences = [[1, 2, 3]]
        with pytest.raises(ValueError, match='Padding type "middle" not understood'):
            pad_sequences(sequences, maxlen=2, padding="middle")


class TestPadSequencesMixed:
    def test_padding_pre_with_truncating_post(self):
        sequences = [[1, 2, 3, 4, 5], [1, 2]]
        result = pad_sequences(sequences, maxlen=3, truncating="post")
        expected = np.array([[1, 2, 3], [0, 1, 2]], dtype="int32")
        np.testing.assert_array_equal(result, expected)

    def test_padding_post_with_truncating_pre(self):
        sequences = [[1, 2, 3, 4, 5], [1, 2]]
        result = pad_sequences(
            sequences, maxlen=3, padding="post", truncating="pre"
        )
        expected = np.array([[3, 4, 5], [1, 2, 0]], dtype="int32")
        np.testing.assert_array_equal(result, expected)

    def test_padding_post_with_truncating_post(self):
        sequences = [[1, 2, 3, 4, 5], [1, 2]]
        result = pad_sequences(
            sequences, maxlen=3, padding="post", truncating="post"
        )
        expected = np.array([[1, 2, 3], [1, 2, 0]], dtype="int32")
        np.testing.assert_array_equal(result, expected)

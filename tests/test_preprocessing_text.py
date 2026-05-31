import pytest
from neutro.preprocessing.text import Tokenizer


class TestTokenizerFitOnTexts:
    def test_fit_on_texts_basic_lowercase(self):
        t = Tokenizer()
        t.fit_on_texts(["Hello World", "Hello Keras"])
        assert t.word_index == {"hello": 1, "world": 2, "keras": 3}
        assert t.index_word[1] == "hello"
        assert t.word_counts == {"hello": 2, "world": 1, "keras": 1}

    def test_fit_on_texts_with_oov_token(self):
        t = Tokenizer(oov_token="<OOV>")
        t.fit_on_texts(["cat dog", "cat fish"])
        assert t.word_index["<OOV>"] == 1
        assert t.index_word[1] == "<OOV>"
        assert t.word_index["cat"] == 2
        assert t.word_index["dog"] == 3

    def test_fit_on_texts_num_words_limit(self):
        t = Tokenizer(num_words=2, oov_token="<OOV>")
        t.fit_on_texts(["apple banana cherry", "apple banana date"])
        # oov_token gets index 1, "apple" gets 2, "banana" gets 3
        # num_words=2 means only index < 2 is kept? No — sorted_words is trimmed to 2 words,
        # but oov_token is added outside that. So we get: oov=1, apple=2, banana=3,
        # but sorted_words only had 2 entries. With num_words=2, sorted_words[:2] = [apple, banana]
        # So word_index has: <OOV>:1, apple:2, banana:3
        assert "<OOV>" in t.word_index
        assert "apple" in t.word_index
        assert "banana" in t.word_index
        assert "cherry" not in t.word_index
        assert "date" not in t.word_index

    def test_fit_on_texts_empty_list(self):
        t = Tokenizer()
        t.fit_on_texts([])
        assert t.word_index == {}
        assert t.index_word == {}


class TestTokenizerTextsToSequences:
    def test_texts_to_sequences_basic(self):
        t = Tokenizer()
        t.fit_on_texts(["the cat sat", "the dog ran"])
        seqs = t.texts_to_sequences(["the cat sat"])
        assert seqs == [[1, 2, 3]]

    def test_texts_to_sequences_with_oov(self):
        t = Tokenizer(oov_token="<OOV>")
        t.fit_on_texts(["cat dog fish"])
        seqs = t.texts_to_sequences(["cat bird dog"])
        # cat=2, dog=3, bird is unknown -> oov=1
        assert seqs == [[2, 1, 3]]

    def test_texts_to_sequences_num_words_filters_to_oov(self):
        t = Tokenizer(num_words=2, oov_token="<OOV>")
        t.fit_on_texts(["apple banana cherry", "apple banana date"])
        seqs = t.texts_to_sequences(["apple banana cherry"])
        # <OOV>=1, apple=2, banana=3
        # num_words=2, so indices >= 2 are filtered to OOV
        # apple (2) >= 2 -> OOV, banana (3) >= 2 -> OOV, cherry unknown -> OOV
        assert seqs == [[1, 1, 1]]

    def test_texts_to_sequences_empty_words(self):
        t = Tokenizer()
        t.fit_on_texts(["hello world"])
        seqs = t.texts_to_sequences(["  hello   world  "])
        assert seqs == [[1, 2]]


class TestTokenizerSequencesToTexts:
    def test_sequences_to_texts_basic(self):
        t = Tokenizer()
        t.fit_on_texts(["hello world"])
        texts = t.sequences_to_texts([[1, 2]])
        assert texts == ["hello world"]

    def test_sequences_to_texts_unknown_index(self):
        t = Tokenizer()
        t.fit_on_texts(["hello world"])
        texts = t.sequences_to_texts([[1, 999]])
        assert texts == ["hello ?"]


class TestTokenizerGetConfig:
    def test_get_config_returns_all_keys(self):
        t = Tokenizer(num_words=10, oov_token="<OOV>", lower=False)
        t.fit_on_texts(["hello world"])
        config = t.get_config()
        assert config["num_words"] == 10
        assert config["oov_token"] == "<OOV>"
        assert config["lower"] is False
        assert config["split"] == " "
        assert "filters" in config
        assert "word_index" in config
        assert "index_word" in config


class TestTokenizerNoLowercase:
    def test_fit_on_texts_without_lowercase(self):
        t = Tokenizer(lower=False)
        t.fit_on_texts(["Hello World"])
        assert "Hello" in t.word_index
        assert "hello" not in t.word_index
        assert "World" in t.word_index

    def test_texts_to_sequences_without_lowercase(self):
        t = Tokenizer(lower=False)
        t.fit_on_texts(["Hello World"])
        seqs = t.texts_to_sequences(["Hello World"])
        assert seqs == [[1, 2]]
        seqs_mismatch = t.texts_to_sequences(["hello world"])
        assert seqs_mismatch == [[]]

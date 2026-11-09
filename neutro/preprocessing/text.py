import re


class Tokenizer:
    """Text tokenization utility class."""

    def __init__(self, num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True, split=' ', oov_token=None):
        self.num_words = num_words
        self.filters = filters
        self.lower = lower
        self.split = split
        self.oov_token = oov_token
        self.word_index = {}
        self.index_word = {}
        self.word_counts = {}

    def _tokenize(self, text):
        if self.lower:
            text = text.lower()
        text = re.sub(f'[{re.escape(self.filters)}]', '', text)
        return [word for word in text.split(self.split) if word]

    def _oov_index(self):
        return self.word_index.get(self.oov_token) if self.oov_token else None

    def _word_index(self, word):
        index = self.word_index.get(word)
        if index is None:
            return self._oov_index()
        if self.num_words and index >= self.num_words:
            return self._oov_index()
        return index

    def fit_on_texts(self, texts):
        for text in texts:
            for word in self._tokenize(text):
                self.word_counts[word] = self.word_counts.get(word, 0) + 1

        words = sorted(self.word_counts.items(), key=lambda item: item[1], reverse=True)
        if self.num_words:
            words = words[:self.num_words]

        start = 1
        if self.oov_token:
            self.word_index[self.oov_token] = 1
            self.index_word[1] = self.oov_token
            start = 2

        for offset, (word, _) in enumerate(words):
            index = offset + start
            self.word_index[word] = index
            self.index_word[index] = word

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = []
            for word in self._tokenize(text):
                index = self._word_index(word)
                if index is not None:
                    sequence.append(index)
            sequences.append(sequence)
        return sequences

    def sequences_to_texts(self, sequences):
        return [
            self.split.join(self.index_word.get(index, '?') for index in sequence)
            for sequence in sequences
        ]

    def get_config(self):
        return {
            'num_words': self.num_words,
            'filters': self.filters,
            'lower': self.lower,
            'split': self.split,
            'oov_token': self.oov_token,
            'word_index': self.word_index,
            'index_word': self.index_word,
        }

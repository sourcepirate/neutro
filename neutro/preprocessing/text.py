import re

class Tokenizer:
    """
    Text tokenization utility class.
    
    Args:
        num_words: The maximum number of words to keep, based on word frequency.
        filters: A string where each element is a character that will be filtered from the texts.
        lower: Whether to convert the texts to lowercase.
        split: The separator used for word splitting.
        oov_token: If given, it will be added to word_index and used to replace out-of-vocabulary words during text_to_sequence calls.
    """
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

    def fit_on_texts(self, texts):
        """
        Updates internal vocabulary based on a list of texts.
        """
        for text in texts:
            if self.lower:
                text = text.lower()
            
            # Filter punctuation
            text = re.sub(f'[{re.escape(self.filters)}]', '', text)
            
            words = text.split(self.split)
            for word in words:
                if not word:
                    continue
                self.word_counts[word] = self.word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        if self.num_words:
            sorted_words = sorted_words[:self.num_words]
            
        start_index = 1
        if self.oov_token:
            self.word_index[self.oov_token] = 1
            self.index_word[1] = self.oov_token
            start_index = 2
            
        for i, (word, _) in enumerate(sorted_words):
            index = i + start_index
            self.word_index[word] = index
            self.index_word[index] = word

    def texts_to_sequences(self, texts):
        """
        Transforms each text in texts to a sequence of integers.
        """
        sequences = []
        for text in texts:
            if self.lower:
                text = text.lower()
            text = re.sub(f'[{re.escape(self.filters)}]', '', text)
            words = text.split(self.split)
            
            seq = []
            for word in words:
                if not word:
                    continue
                index = self.word_index.get(word)
                if index is not None:
                    if self.num_words and index >= self.num_words:
                        if self.oov_token:
                            seq.append(self.word_index[self.oov_token])
                    else:
                        seq.append(index)
                elif self.oov_token:
                    seq.append(self.word_index[self.oov_token])
            sequences.append(seq)
        return sequences

    def sequences_to_texts(self, sequences):
        """
        Transforms each sequence of integers to text.
        """
        texts = []
        for seq in sequences:
            words = [self.index_word.get(i, '?') for i in seq]
            texts.append(self.split.join(words))
        return texts
    
    def get_config(self):
        return {
            'num_words': self.num_words,
            'filters': self.filters,
            'lower': self.lower,
            'split': self.split,
            'oov_token': self.oov_token,
            'word_index': self.word_index,
            'index_word': self.index_word
        }

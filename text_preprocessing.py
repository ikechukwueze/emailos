import string
import spacy
import contractions
from nltk.corpus import stopwords as nltk_stopwords
import pickle
import numpy as np
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences






class TextPreprocessor:
    def __init__(self):
        """
        Initialize the TextPreprocessor class.
        Load required NLP resources and sets for preprocessing.
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.punctuations = set(string.punctuation)
        self.stopwords = set(nltk_stopwords.words("english"))


    def normalize_text(self, text):
        """
        Normalize text by removing leading/trailing spaces and converting to lowercase.
        
        Args:
            text (str): Input text to be normalized.

        Returns:
            str: Normalized text.
        """
        return text.strip().lower()


    def expand_contractions(self, text):
        """
        Expand contractions in the text using the `contractions` library.
        
        Args:
            text (str): Input text with contractions.

        Returns:
            str: Text with expanded contractions.
        """
        return contractions.fix(text)


    def remove_punctuations(self, text):
        """
        Remove punctuation marks from the text.
        
        Args:
            text (str): Input text with punctuation marks.

        Returns:
            str: Text with punctuations removed.
        """
        text_wo_punctuations = ""
        for char in text:
            if char in self.punctuations:
                text_wo_punctuations += " "
            else:
                text_wo_punctuations += char
        return text_wo_punctuations.strip()


    def tokenize_text(self, text):
        """
        Tokenize the input text using the spaCy NLP library.
        
        Args:
            text (str): Input text to be tokenized.

        Returns:
            list: List of tokens.
        """
        doc = self.nlp(text)
        return [token.text for token in doc]


    def remove_stopwords(self, text, exclude=[], include=[]):
        """
        Remove stopwords from the text.
        
        Args:
            text (str): Input text with stopwords.
            exclude (list): List of stopwords to exclude.
            include (list): List of words to include as stopwords.

        Returns:
            str: Text with stopwords removed.
        """
        doc = self.nlp(text)

        stopwords = self.stopwords - set(exclude)
        stopwords = stopwords.union(set(include))

        tokens = [token.text for token in doc if not token.text in stopwords]
        text_wo_stopwords = " ".join(tokens)
        return text_wo_stopwords


    def lemmatize_text(self, text):
        """
        Lemmatize the input text using the spaCy NLP library.
        
        Args:
            text (str): Input text to be lemmatized.

        Returns:
            str: Text with lemmatized tokens.
        """
        doc = self.nlp(text)
        lemmatized_tokens = [token.lemma_ for token in doc]
        lemmatized_text = " ".join(lemmatized_tokens)
        return lemmatized_text


    def process_text(self, text, include_stopwords=[], exclude_stopwords=[]):
        """
        Process text through all preprocessing steps.
        
        Args:
            text (str): Input text to be preprocessed.
            include_stopwords (list): Additional words to include as stopwords.
            exclude_stopwords (list): Words to exclude from stopwords.

        Returns:
            str: Processed text after all preprocessing steps.
        """
        text = self.normalize_text(text)
        text = self.expand_contractions(text)
        text = self.remove_punctuations(text)
        text = self.remove_stopwords(text, exclude_stopwords, include_stopwords)
        text = self.lemmatize_text(text)
        return text






# class CustomTextVectorizer:
#     """
#     Custom text vectorizer class for vectorizing text sequences.

#     Attributes:
#     - vocab_size (int): The size of the vocabulary.
#     - max_tokens (int): The maximum number of tokens in a text sequence.
#     - trunc_type (str): The truncation type for sequences ("pre" or "post").
#     - padding_type (str): The padding type for sequences ("pre" or "post").
#     - oov_token (str): The token to represent out-of-vocabulary words.
#     - text_sequence (list): List of text sequences to tokenize.
#     - tokenizer (Tokenizer): Tokenizer object for tokenization.

#     Methods:
#     - get_max_num_of_tokens_in_text_sequence: Returns the maximum number of tokens in a text sequence.
#     - fit_on_texts: Fits the tokenizer on the text sequences and sets vocab_size and max_tokens.
#     - texts_to_sequences: Converts text sequences to integer sequences using the tokenizer.

#     """

#     def __init__(self):
#         """
#         Initializes a CustomTextVectorizer object.

#         """
#         self.vocab_size = None
#         self.max_tokens = 50
#         self.trunc_type = "post"
#         self.padding_type = "post"
#         self.oov_token = "<OOV>"
#         self.tokenizer = Tokenizer(oov_token=self.oov_token)

#     def get_vocabulary(self):
#         return list(self.tokenizer.word_index.keys())

#     def fit(self, text_sequence):
#         """
#         Fits the tokenizer on the text sequences and sets vocab_size and max_tokens.

#         """
#         self.tokenizer.fit_on_texts(text_sequence)
#         self.vocab_size = len(self.tokenizer.word_index) + 1

#     def transform(self, text_sequence):
#         """
#         Converts text sequences to integer sequences using the tokenizer.

#         Returns:
#         - integer_sequences (ndarray): Integer sequences representing the text sequences.

#         """
#         integer_sequences = self.tokenizer.texts_to_sequences(text_sequence)
#         integer_sequences = pad_sequences(
#             integer_sequences,
#             maxlen=self.max_tokens,
#             padding=self.padding_type,
#             truncating=self.trunc_type,
#         )
#         return np.array(integer_sequences)

#     def fit_transform(self, text_sequence):
#         self.fit(text_sequence)
#         return self.transform(text_sequence)



# def create_glove_embedding_matrix(vocabulary, path_to_glove_pkl, embedding_size):
#     vocab_size = len(vocabulary) + 1
#     vocabulary_indices = [i for i in range(vocab_size)]
#     word_index = dict(zip(vocabulary, vocabulary_indices))

#     with open(path_to_glove_pkl, "rb") as file:
#         glove_token_to_vector = pickle.load(file)

#     embedding_matrix = np.zeros((vocab_size, embedding_size))
#     for word, i in word_index.items():
#         embedding_vector = glove_token_to_vector.get(word)
#         if embedding_vector is not None:
#             embedding_matrix[i] = embedding_vector

#     return embedding_matrix




def convert_sentiment_to_integer(sentiment):
    """
    Convert sentiment label to an integer value.
    
    Args:
        sentiment (str): Sentiment label (Negative/Positive).

    Returns:
        int: Integer representation of the sentiment (0 for Negative, 1 for Positive).
    """
    sentiment_mapping = {"Negative": 0, "Positive": 1}
    sentiment = sentiment.strip()
    return sentiment_mapping[sentiment]


def convert_integer_to_sentiment(integer):
    """
    Convert integer value to an sentiment label
    
    Args:
        integer (int): integer value (1/o).

    Returns:
        str: Sentiment representation
    """
    sentiment_mapping = {0: "Negative", 1: "Positive"}
    return sentiment_mapping[integer]
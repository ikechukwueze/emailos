import string
import spacy
import contractions
from nltk.corpus import stopwords as nltk_stopwords






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
        return text_wo_stopwords.strip()


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
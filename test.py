import unittest
from text_preprocessing import TextPreprocessor




class TestTextPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = TextPreprocessor()

    def test_normalize_text(self):
        self.assertEqual(self.preprocessor.normalize_text("Hello World"), "hello world")

    def test_expand_contractions(self):
        self.assertEqual(self.preprocessor.expand_contractions("I'll be there."), "I will be there.")

    def test_remove_punctuations(self):
        self.assertEqual(self.preprocessor.remove_punctuations("Hello, world!"), "Hello  world")

    def test_tokenize_text(self):
        tokens = self.preprocessor.tokenize_text("Hello world")
        self.assertEqual(tokens, ["Hello", "world"])

    def test_remove_stopwords(self):
        text = "This is a sentence."
        processed_text = self.preprocessor.remove_stopwords(text)
        self.assertEqual(processed_text, "This sentence .")

    def test_lemmatize_text(self):
        lemmatized_text = self.preprocessor.lemmatize_text("Going up the stairs")
        self.assertEqual(lemmatized_text, "go up the stair")

    def test_process_text(self):
        processed_text = self.preprocessor.process_text("I'm running up the stairs.")
        self.assertEqual(processed_text, "run stair")

if __name__ == '__main__':
    unittest.main()



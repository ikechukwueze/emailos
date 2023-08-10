import os
import pandas as pd
from flask import Flask, request, jsonify
from text_preprocessing import TextPreprocessor, convert_integer_to_sentiment
from helpers import unpickle_python_object




app = Flask(__name__)
project_path = os.getcwd()

# Paths to the saved model and transformer
model_path = os.path.join(project_path, "models", "ensemble_LR-SVC-GNB_model.pkl")
transformer_path = os.path.join(project_path, "models", "feature_transformer.pkl")

# Load the saved model and transformer
model = unpickle_python_object(model_path)
transformer = unpickle_python_object(transformer_path)

# Initialize the text processor
review_processor = TextPreprocessor()







@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment based on input data.

    This route expects a JSON request containing review, rating, and gender.
    It processes the review text, transforms it, makes a prediction using the model,
    and returns the predicted sentiment.

    Returns:
        JSON response with predicted sentiment or an error message.
    """
    try:
        # Get input data from the JSON request
        data = request.get_json()
        review = data['review']
        rating = int(data['rating'])
        gender = data['gender']
        
        # Preprocess the review text
        processed_review = review_processor.process_text(review)
        
        # Prepare data for prediction
        data = {'Processed_review': [processed_review], 'Rating': [rating], 'Gender': [gender]}
        df = pd.DataFrame.from_dict(data)
        
        # Transform data and make a prediction
        features = transformer.transform(df).toarray()
        result = model.predict(features)
        
        # Convert prediction to sentiment and create response
        response = convert_integer_to_sentiment(result.item())
        return jsonify(response), 200

    except Exception as e:
        error_response = {'error': 'An error occurred', 'details': str(e)}
        return jsonify(error_response), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

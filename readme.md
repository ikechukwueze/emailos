
---

# Sentiment Prediction API

This repository contains a Flask app that exposes an API endpoint for predicting product sentiment based on review, rating, and gender.

## Prerequisites

- Docker: Make sure you have Docker installed on your machine. If not, you can download it from [here](https://www.docker.com/get-started).

## Getting Started

Follow these steps to set up and run the Flask app with Docker:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/ikechukwueze/emailos.git
   cd emailos
   ```

2. **Build and Run the Docker Image:**

   Build the Docker image and run the app in a Docker container:

   ```bash
   docker-compose up --build
   ```

   This will build the image and start the container. The Flask app will be accessible at `http://localhost:80`.

3. **API Endpoint:**

   The API endpoint for sentiment prediction will be available at:

   ```
   http://localhost:80/predict
   ```

   Send a POST request to this endpoint with JSON data containing the review, rating, and gender to get the sentiment prediction.

   Example JSON data:
   ```json
   {
       "review": "This product is amazing!",
       "rating": 5,
       "gender": "Female"
   }
   ```

4. **Stop the App:**

   To stop the app and the Docker container, press `Ctrl + C` in the terminal.

5. **Cleanup:**

   After stopping the app, you can remove the Docker containers and images associated with this app using:

   ```bash
   docker-compose down
   ```
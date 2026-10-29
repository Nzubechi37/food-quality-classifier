# Food Quality Classifier

A containerized machine learning application that classifies fruit images as fresh or rotten using PyTorch, FastAPI, Streamlit, and Docker.

## Features
- Fresh vs rotten image classification
- FastAPI prediction endpoint
- Streamlit web interface
- Docker Compose deployment

## Run the Application
Run `docker compose up --build` from the project directory.

Web app: http://localhost:8501
API documentation: http://localhost:8000/docs
Health check: http://localhost:8000/health

## Model Evaluation
The existing model previously achieved 88.12% test accuracy. Results on new images may differ.

The original dataset is excluded from version control. The existing trained model is used, so retraining is not required.

## Live Demo
Try the application here: https://food-quality-classifier-oraz.onrender.com


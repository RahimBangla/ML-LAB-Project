#!/bin/bash

# Login to the GitHub Container Registry
echo "Logging in to GitHub Container Registry..."
# echo "$GITHUB_TOKEN" | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin

# Pull Docker images for Heart Disease Prediction project
echo "Pulling the backend image..."
docker pull ghcr.io/rahimbangla/heart-disease-prediction-backend:main

echo "Pulling the frontend image..."
docker pull ghcr.io/rahimbangla/heart-disease-prediction-frontend:main

# Stop and remove existing containers
docker-compose down

# Start Docker Compose
echo "Starting Docker containers using docker-compose..."
docker-compose up -d

echo "Heart Disease Prediction containers have been started successfully."

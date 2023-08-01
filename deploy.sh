#!/bin/bash

set -e

DEPLOY_TYPE=$1
TMP_DEPLOY_DIR=deploy

# Check if the deployment type argument is provided
if [ -z "$DEPLOY_TYPE" ]; then
  echo "Error: Please provide the deployment type argument ('dev' or 'prod')."
  echo "Usage: $0 <deployment_type>"
  exit 1
fi

# Print the current branch and commit hash
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
CURRENT_COMMIT=$(git rev-parse HEAD)
echo "Deploying from branch: $CURRENT_BRANCH"
echo "Commit hash: $CURRENT_COMMIT"

# Clean up the previous deployment
rm -rf $TMP_DEPLOY_DIR
mkdir -p $TMP_DEPLOY_DIR
mkdir -p $TMP_DEPLOY_DIR/src

# Add requirements file
cp requirements.txt $TMP_DEPLOY_DIR/

# Add source code needed
cp src/*.py src/sample_questions.csv $TMP_DEPLOY_DIR/src/

# Change directory to $TMP_DEPLOY_DIR folder
cd $TMP_DEPLOY_DIR

# Create the README configuration (necessary for Hugging Face space)
echo '---
title: Buster Dev
emoji: 💻
colorFrom: pink
colorTo: green
sdk: gradio
sdk_version: 3.39.0
app_file: src/buster_app.py
python: 3.11
pinned: false
---' > README.md

# Additional actions for dev deployment, if needed
if [ "$DEPLOY_TYPE" = "dev" ]; then
  echo "Performing dev-specific actions..."
  # Add any specific actions for dev here
fi

# Additional actions for prod deployment, if needed
if [ "$DEPLOY_TYPE" = "prod" ]; then
  echo "Performing prod-specific actions..."
  # Add any specific actions for prod here
fi

# Initialize a new, temporary git repository, add the files, and commit them
git init
git add -A
git commit -m "Deploy app"

# Set the remote URL based on the deployment type
if [ "$DEPLOY_TYPE" = "dev" ]; then
  # Replace <YOUR-ORG> and <YOUR-DEV-SPACE> with the actual values for your Hugging Face organization and dev space
  git remote add space https://huggingface.co/spaces/databank-ai4h/buster-dev
elif [ "$DEPLOY_TYPE" = "prod" ]; then
  # Replace <YOUR-ORG> and <YOUR-PROD-SPACE> with the actual values for your Hugging Face organization and prod space
  git remote add space https://huggingface.co/spaces/databank-ai4h/buster-prod
fi

# Push all files to the Hugging Face space
git push --force space main

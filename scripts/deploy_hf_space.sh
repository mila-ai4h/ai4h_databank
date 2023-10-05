#!/bin/bash

set -e

# User Args
DEPLOY_TYPE=$1
TMP_DEPLOY_DIR=${2:-"deploy"}


# SPACE URLS
DEV_URL="https://$HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/databank-ai4h/buster-dev"
PROD_URL="https://$HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/databank-ai4h/buster-prod"

# Authenticate to huggingface for push
huggingface-cli login --token $HF_TOKEN

# Check and set deployment URL
set_deploy_url() {
  if [ -z "$DEPLOY_TYPE" ]; then
    echo "Error: Please provide the deployment type argument ('dev' or 'prod')."
    echo "Usage: $0 <deployment_type> [deploy_directory]"
    exit 1
  elif [ "$DEPLOY_TYPE" = "dev" ]; then
    SPACE_URL=$DEV_URL
    echo "Deploying to **dev** space"
  elif [ "$DEPLOY_TYPE" = "prod" ]; then
    SPACE_URL=$PROD_URL
    echo "Deploying to **prod** space"
  else
    echo "Error: Invalid DEPLOY_TYPE. Valid values are 'dev' or 'prod'."
    exit 1
  fi
}

# Check and set deployment directory
set_deploy_dir() {
  if [ -z "$TMP_DEPLOY_DIR" ]; then
    TMP_DEPLOY_DIR="deploy"
  fi
  echo "Using deployment directory: $TMP_DEPLOY_DIR"
}

# Print current branch and commit details
print_git_details() {
  CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
  CURRENT_COMMIT=$(git rev-parse HEAD)
  echo "Deploying from branch: $CURRENT_BRANCH"
  echo "Commit hash: $CURRENT_COMMIT"
}

# Prepare deployment directory
prepare_deploy_dir() {
  rm -rf $TMP_DEPLOY_DIR
  mkdir -p $TMP_DEPLOY_DIR/src
  mkdir -p $TMP_DEPLOY_DIR/data
  mkdir -p $TMP_DEPLOY_DIR/assets

  # Copy over required files
  cp requirements.txt $TMP_DEPLOY_DIR/
  cp -r src/buster/  $TMP_DEPLOY_DIR/src/buster/
  cp src/feedback.py src/cfg.py src/app_utils.py $TMP_DEPLOY_DIR/src/
  cp data/documents_metadata.csv $TMP_DEPLOY_DIR/data
  cp src/buster/assets/index.html $TMP_DEPLOY_DIR/assets/index.html

  # make folder cwd
  cd $TMP_DEPLOY_DIR
}

# Create README for Hugging Face space
create_readme() {
  echo '---
title: Buster
emoji: 💻
colorFrom: pink
colorTo: green
sdk: gradio
app_file: src/buster/gradio_app.py
python: 3.11
pinned: false
---' > README.md
}

# Initialize and push to git
git_operations() {
  git config --global init.defaultBranch main
  git init
  git branch -m main
  git config user.email "jerpint@gmail.com"
  git config user.name "Jeremy Pinto"
  git add -A
  git commit -m "Deploy app"
  git remote add space $SPACE_URL
  git push --force space main
}

# Main execution
set_deploy_url
set_deploy_dir
print_git_details
prepare_deploy_dir
create_readme
git_operations

#!/bin/bash

# Check if conda environment already exists
if [ -z "$(conda info --envs | grep video_analysis)" ]; then
  # Step 1: Create a new conda environment
  conda create --name video_analysis python=3.10.15 -y
fi

# Step 2: Activate the environment and install requirements
conda activate video_analysis
pip install -r requirements.txt

# Step 3: Create an empty json file and directories
if [ ! -f ai-hand-service-acc.json ]; then
  touch ai-hand-service-acc.json
fi

if [ ! -d recordings ]; then
  mkdir -p recordings
fi

if [ ! -d outputs_json ]; then
  mkdir -p outputs_json
fi

# Welcome to the VideoAnalysisPipeline

### This project can be built simply by running the install.sh script




## C+P the following in your bash terminal to build this project

```bash
#!/bin/bash

# Step 1: Create a new conda environment
conda create --name video_analysis python=3.10.15 -y

# Step 2: Activate the environment and install requirements
conda activate video_analysis
pip install -r requirements.txt

# Step 3: Create an empty json file and directories
touch ai-hand-service-acc.json
mkdir -p recordings outputs_json
```
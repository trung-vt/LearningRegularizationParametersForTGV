#!/bin/bash

# Step 1: Define environment variables
export MY_VAR1="value1"
export MY_VAR2="value2"
export PATH="/my/custom/path:$PATH"
export LD_LIBRARY_PATH="/my/custom/path:$LD_LIBRARY_PATH"

# Step 2: Source the Python virtual environment
# Replace the path below with the actual path to your virtual environment's 'activate' script
# source /path/to/your/venv/bin/activate
source venv/bin/activate

# Optional: Deactivate conda
conda deactivate

# Optional: Print a message to confirm the environment is set
echo "Environment variables are set, and the virtual environment is activated."

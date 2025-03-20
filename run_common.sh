#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Set environment name
ENV_NAME="auto_llm_eval"

# echo "==== DEBUG INFORMATION ===="
# echo "CONDA_DEFAULT_ENV = $CONDA_DEFAULT_ENV"
# echo "CONDA_PREFIX = $CONDA_PREFIX"
# echo "Current Python: $(which python)"
# echo "Current PATH: $PATH"
# echo "Current shell: $SHELL"
# echo "Conda environments:"
# conda env list

# More robust detection using multiple methods
if [[ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]]; then
    ACTIVE_ENV_METHOD1="YES"
else
    ACTIVE_ENV_METHOD1="NO"
fi

ACTIVE_ENV_NAME=$(conda info --envs | grep '*' | awk '{print $1}')
if [[ "$ACTIVE_ENV_NAME" == "$ENV_NAME" ]]; then
    ACTIVE_ENV_METHOD2="YES"
else
    ACTIVE_ENV_METHOD2="NO"
fi

PYTHON_PATH=$(which python)
if [[ "$PYTHON_PATH" == *"/envs/$ENV_NAME/"* ]]; then
    ACTIVE_ENV_METHOD3="YES"
else
    ACTIVE_ENV_METHOD3="NO"
fi

# echo "Is environment active (method 1 - CONDA_DEFAULT_ENV): $ACTIVE_ENV_METHOD1"
# echo "Is environment active (method 2 - conda info): $ACTIVE_ENV_METHOD2"
# echo "Is environment active (method 3 - Python path): $ACTIVE_ENV_METHOD3"
# echo "============================="

# Initialize conda for script
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if environment exists, create if it doesn't
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "Creating conda environment: $ENV_NAME"
    conda env create -f "$SCRIPT_DIR/environment.yml"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create conda environment"
        exit 1
    fi
fi

# Try all methods to detect if environment is active
if [[ "$ACTIVE_ENV_METHOD1" == "YES" || "$ACTIVE_ENV_METHOD2" == "YES" || "$ACTIVE_ENV_METHOD3" == "YES" ]]; then
    echo "Using existing conda environment: $ENV_NAME"
else
    echo "Activating conda environment: $ENV_NAME"
    conda activate $ENV_NAME
    if [ $? -ne 0 ]; then
        echo "Error: Failed to activate conda environment"
        exit 1
    fi
fi

# Print post-activation state
# echo "After activation check: Current Python = $(which python)"

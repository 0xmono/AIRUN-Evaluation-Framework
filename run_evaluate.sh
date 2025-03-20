#!/bin/bash

source run_common.sh

# Add a directory to PATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Now you can run your Python script
python evaluate.py --data-dir data/sample --scenarios 2 --report-path data/sample/grades.csv
#--data-dir data/sample --scenarios 2 --report-path data/sample/grades.csv

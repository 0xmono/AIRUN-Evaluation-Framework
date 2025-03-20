#!/bin/bash

source run_common.sh

# Add `src` directory to PATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Default test directory
DEFAULT_TEST_PATH="tests/epam/auto_llm_eval"

# Set test path: use first argument if provided, otherwise use default
TEST_PATH=${1:-$DEFAULT_TEST_PATH}

# Set up colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print information about the test run
echo -e "${YELLOW}Running pytest on:${NC} $TEST_PATH"

# Run the tests with verbose output and print statements
# Add any other pytest options you commonly use
pytest $TEST_PATH -v -s

# Get the exit code from pytest
EXIT_CODE=$?

# Print completion message
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${YELLOW}Tests completed with failures.${NC}"
fi

# Exit with the same code as pytest
exit $EXIT_CODE

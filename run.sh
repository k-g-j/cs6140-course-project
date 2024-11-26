#!/bin/bash

# run.sh - Main script to execute the renewable energy prediction project

# Print header
echo "========================================="
echo "Running Renewable Energy Prediction Project"
echo "========================================="

# Create required directories if they don't exist
echo "Creating required directories..."
mkdir -p data
mkdir -p processed_data
mkdir -p models
mkdir -p figures
mkdir -p logs

# Check if python environment is set up
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if requirements are installed
if [ ! -f "requirements.txt" ]; then
    echo "requirements.txt not found!"
    exit 1
fi

# Install dependencies if needed
echo "Installing/updating dependencies..."
pip install -r requirements.txt

# Run data processing pipeline
echo -e "\nStep 1: Running data processing pipeline..."
python3 src/main.py

# Run model training
echo -e "\nStep 2: Running model training..."
python3 src/models/train.py

# Run model evaluation
echo -e "\nStep 3: Running model evaluation..."
python3 src/models/evaluate.py

# Run ablation studies
echo -e "\nStep 4: Running ablation studies..."
python3 run_ablation.py

# Check for errors in log files
echo -e "\nChecking logs for errors..."
if grep -i "error" logs/*.log &> /dev/null; then
    echo "WARNING: Errors found in logs. Please check log files in logs directory."
else
    echo "No errors found in logs."
fi

# Print summary
echo -e "\n========================================="
echo "Project execution completed!"
echo "----------------------------------------"
echo "Generated files can be found in:"
echo "- Processed data: ./processed_data/"
echo "- Model outputs: ./models/"
echo "- Figures: ./figures/"
echo "- Logs: ./logs/"
echo "========================================="

# Optional: Generate timestamp for records
echo "Completed at: $(date)" >> logs/execution_history.log
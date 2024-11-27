#!/bin/bash

# run.sh - Main script to execute the renewable energy prediction project

# Exit on error
set -e

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
mkdir -p figures/final_analysis
mkdir -p figures/feature_analysis
mkdir -p analysis_results
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

# Function to check last command's status
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed"
        exit 1
    fi
}

# Install dependencies if needed
echo -e "\nInstalling/updating dependencies..."
pip install -r requirements.txt
check_status "Dependencies installation"

# Run data preprocessing pipeline
echo -e "\nStep 1: Running data preprocessing pipeline..."
python3 main.py
check_status "Data preprocessing"

# Run feature engineering
echo -e "\nStep 2: Running feature engineering..."
python3 src/data/feature_engineering.py
check_status "Feature engineering"

# Run initial model training
echo -e "\nStep 3: Running initial model training..."
python3 run.py
check_status "Initial model training"

# Run model evaluation
echo -e "\nStep 4: Running model evaluation..."
python3 src/models/evaluate.py
check_status "Model evaluation"

# Run ablation studies
echo -e "\nStep 5: Running ablation studies..."
python3 run_ablation.py
check_status "Ablation studies"

# Run results analysis
echo -e "\nStep 6: Running results analysis..."
python3 analyze_results.py
check_status "Results analysis"

# Run model refinement
echo -e "\nStep 7: Running model refinement..."
python3 refine_models.py
check_status "Model refinement"

# Create final visualizations
echo -e "\nStep 8: Creating final visualizations..."
python3 create_visualizations.py
check_status "Visualization creation"

# Generate final report
echo -e "\nStep 9: Generating final report..."
python3 analyze_results.py
check_status "Report generation"

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
echo "- Analysis results: ./analysis_results/"
echo "- Final visualizations: ./figures/final_analysis/"
echo "- Feature analysis: ./figures/feature_analysis/"
echo "- Logs: ./logs/"
echo "========================================="

# Record execution timestamp
echo "Completed at: $(date)" >> logs/execution_history.log

# Final status check
echo -e "\nChecking final status..."
if [ -f "models/training_results.yaml" ] && \
   [ -f "figures/final_analysis/model_comparison.png" ] && \
   [ -f "analysis_results/analysis_report.md" ]; then
    echo "All critical outputs generated successfully!"
else
    echo "WARNING: Some expected outputs are missing. Please check the logs."
fi
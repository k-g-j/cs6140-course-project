#!/bin/bash

# run.sh - Main script to execute the renewable energy prediction project

# Exit on error
set -e

# Print header
echo "========================================="
echo "Running Renewable Energy Prediction Project"
echo "========================================="

# Set up Python path to include src directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

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

# Function to check last command's status and data files
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed"
        exit 1
    fi
}

check_data_file() {
    if [ ! -f "$1" ]; then
        echo "Error: Required data file $1 not found"
        exit 1
    fi
}

# Install dependencies if needed
echo -e "\nInstalling/updating dependencies..."
pip install -r requirements.txt
check_status "Dependencies installation"

# Run data preprocessing pipeline
echo -e "\nStep 1: Running data preprocessing pipeline..."
python3 src/main.py
check_status "Data preprocessing"
check_data_file "processed_data/final_processed_data.csv"

# Run feature engineering
echo -e "\nStep 2: Running feature engineering..."
python3 src/data/feature_engineering.py
check_status "Feature engineering"
check_data_file "processed_data/engineered_features.csv"

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
echo -e "\nStep 9: Generating final report and visualizations..."
jupyter nbconvert --execute notebooks/*.ipynb --to pdf
check_status "Notebook conversion"

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
echo "- Notebooks (PDF): ./notebooks/"
echo "========================================="

# Record execution timestamp
echo "Completed at: $(date)" >> logs/execution_history.log

# Final status check
echo -e "\nChecking final status..."
required_files=(
    "processed_data/final_processed_data.csv"
    "processed_data/engineered_features.csv"
    "models/training_results.yaml"
    "figures/final_analysis/model_comparison.png"
    "analysis_results/analysis_report.md"
)

missing_files=0
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "WARNING: Missing expected output file: $file"
        missing_files=$((missing_files + 1))
    fi
done

if [ $missing_files -eq 0 ]; then
    echo "All critical outputs generated successfully!"
else
    echo "WARNING: $missing_files expected output files are missing. Check the logs for errors."
fi
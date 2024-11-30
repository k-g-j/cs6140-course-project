#!/bin/bash

# Exit on error
set -e

# Print header
echo "========================================="
echo "Running Renewable Energy Prediction Project"
echo "========================================="

# Function to check command status
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed"
        exit 1
    fi
}

# Function to check dependencies
check_dependencies() {
    local missing=0

    # Check for required LaTeX packages
    echo "Checking required dependencies..."

    # Check for Python and pip
    if ! command -v python3 &> /dev/null; then
        echo "Error: python3 is required but not installed"
        missing=$((missing + 1))
    fi

    if ! command -v pip &> /dev/null; then
        echo "Error: pip is required but not installed"
        missing=$((missing + 1))
    fi

    # Check for Jupyter
    if ! command -v jupyter &> /dev/null; then
        echo "Error: jupyter is required but not installed"
        missing=$((missing + 1))
    fi

    # Check for LaTeX dependencies
    if ! command -v xelatex &> /dev/null; then
        echo "Error: xelatex is required but not installed"
        missing=$((missing + 1))
    fi

    if ! command -v bibtex &> /dev/null; then
        echo "Error: bibtex is required but not installed"
        missing=$((missing + 1))
    fi

    return $missing
}

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo "Error: Required file $1 not found"
        exit 1
    fi
}

# Function to check directory permissions
check_permissions() {
    local dir=$1
    if [ ! -w "$dir" ]; then
        echo "Error: No write permission for $dir"
        exit 1
    fi
}

# Function to cleanup temporary files
cleanup_temp_files() {
    echo "Cleaning up temporary files..."
    rm -f notebook.tex
    rm -f notebook.aux
    rm -f notebook.out
    rm -f notebook.log
    rm -f texput.log
    find . -name "*.aux" -type f -delete
    find . -name "*.log" -type f -delete
    find . -name "*.out" -type f -delete
}

# Function to convert notebooks to PDF
convert_notebooks() {
    local conversion_errors=0

    for notebook in notebooks/*.ipynb; do
        if [ -f "$notebook" ]; then
            echo "Converting $notebook to PDF..."

            # Extract basename for output file
            local basename=$(basename "$notebook" .ipynb)
            local output_pdf="notebooks/${basename}.pdf"

            jupyter nbconvert --to pdf "$notebook" \
                --template=classic \
                --PDFExporter.verbose=True \
                --PDFExporter.latex_count=3 \
                --PDFExporter.template_file=classic \
                --output-dir="notebooks" || {
                    echo "Warning: PDF conversion failed for $notebook"
                    echo "$(date): Failed to convert $notebook to PDF" >> logs/conversion_errors.log
                    conversion_errors=$((conversion_errors + 1))
                    continue
                }

            if [ -f "$output_pdf" ]; then
                echo "Successfully created $output_pdf"
            else
                echo "Warning: PDF file not created for $notebook"
                conversion_errors=$((conversion_errors + 1))
            fi
        fi
    done

    return $conversion_errors
}

# Function to validate outputs
validate_outputs() {
    local missing=0
    declare -a required_files=(
        "processed_data/final_processed_data.csv"
        "processed_data/engineered_features.csv"
        "models/training_results.yaml"
        "models/refined_results.yaml"
        "figures/final_analysis/model_comparison.png"
        "analysis_results/analysis_report.md"
        "figures/ablation_studies/ablation_results.yaml"
        "logs/execution_history.log"
    )

    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            echo "ERROR: Missing required output file: $file"
            echo "Possible reasons: The corresponding step failed or was skipped."
            echo "Check the logs for more details."
            missing=$((missing + 1))
        fi
    done

    # Check if any PDFs were generated
    if ! ls notebooks/*.pdf >/dev/null 2>&1; then
        echo "WARNING: No PDF files were generated from notebooks"
        echo "This may not be critical but should be investigated"
    fi

    return $missing
}

# Check dependencies before starting
check_dependencies
if [ $? -ne 0 ]; then
    echo "Please install missing dependencies before continuing."
    exit 1
fi

# Create required directories
echo "Creating required directories..."
mkdir -p data
mkdir -p processed_data
mkdir -p models
mkdir -p figures/{exploration,feature_analysis,final_analysis,models,ablation_studies}
mkdir -p analysis_results
mkdir -p logs
mkdir -p notebooks
mkdir -p reports

# Check directory permissions
for dir in data processed_data models figures analysis_results logs notebooks reports; do
    check_permissions $dir
done

# Set up Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Install/update dependencies
echo -e "\nInstalling/updating dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
check_status "Dependencies installation"

# Clean up any previous log files older than 7 days
echo -e "\nCleaning up old log files..."
find logs -name "*.log" -type f -mtime +7 -delete

# Start logging
exec 1> >(tee -a "logs/execution_history.log")
exec 2>&1

echo "Starting execution at $(date)"

# Run data preprocessing pipeline
echo -e "\nStep 1: Running data preprocessing pipeline..."
python3 src/main.py
check_status "Data preprocessing"
check_file "processed_data/final_processed_data.csv"

# Run feature engineering
echo -e "\nStep 2: Running feature engineering..."
python3 src/data/feature_engineering.py
check_status "Feature engineering"
check_file "processed_data/engineered_features.csv"

# Run initial model training
echo -e "\nStep 3: Running initial model training..."
python3 run.py
check_status "Initial model training"
check_file "models/training_results.yaml"

# Run model evaluation
echo -e "\nStep 4: Running model evaluation..."
python3 src/models/evaluate.py
check_status "Model evaluation"

# Run ablation studies
echo -e "\nStep 5: Running ablation studies..."
python3 run_ablation.py
check_status "Ablation studies"
check_file "figures/ablation_studies/ablation_results.yaml"

# Run results analysis
echo -e "\nStep 6: Running results analysis..."
python3 analyze_results.py
check_status "Results analysis"
check_file "analysis_results/analysis_report.md"

# Run model refinement
echo -e "\nStep 7: Running model refinement..."
python3 refine_models.py
check_status "Model refinement"
check_file "models/refined_results.yaml"

# Create visualizations
echo -e "\nStep 8: Creating final visualizations..."
python3 create_visualizations.py
check_status "Visualization creation"
check_file "figures/final_analysis/model_comparison.png"

# Generate final report and convert notebooks
echo -e "\nStep 9: Converting notebooks to PDF..."

# Convert notebooks with error handling
if ! convert_notebooks; then
    echo "WARNING: Some notebooks failed to convert to PDF"
    echo "Check logs/conversion_errors.log for details"
else
    echo "Notebook conversion completed successfully"
fi

# Cleanup temporary files
cleanup_temp_files

# Validate all outputs
echo -e "\nValidating outputs..."
validate_outputs
output_status=$?

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

# Record completion timestamp
echo "Completed at: $(date)" >> logs/execution_history.log

# Exit with appropriate status
if [ $output_status -eq 0 ]; then
    echo "All outputs validated successfully!"
    exit 0
else
    echo "WARNING: Some outputs are missing. Check the logs for errors."
    exit 1
fi
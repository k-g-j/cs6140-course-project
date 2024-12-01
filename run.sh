#!/bin/bash

# Exit on error
set -e

# Create logs directory
mkdir -p logs

# Redirect stdout to both console and log file, stderr to error log
exec 1> >(tee -a "logs/execution_history.log")
exec 2> "logs/error.log"

# Suppress warnings
export PYTHONWARNINGS="ignore"
# Suppress sklearn verbose output
export SKLEARN_VERBOSE=0

# Print header
echo "========================================="
echo "Running Renewable Energy Prediction Project"
echo "========================================="

# Function to check command status quietly
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed"
        exit 1
    fi
}

# Function to check dependencies quietly
check_dependencies() {
    local missing=0
    echo "Checking dependencies..."

    # Check for system package managers and dependencies
    if [ "$(uname)" = "Darwin" ]; then
        if ! command -v brew >/dev/null 2>&1; then
            echo "Error: Homebrew required"
            missing=$((missing + 1))
        else
            if ! brew list zeromq >/dev/null 2>&1; then
                brew install zeromq >/dev/null 2>&1
            fi
        fi

        if ! command -v xelatex &> /dev/null; then
            echo "Error: MacTeX required"
            missing=$((missing + 1))
        fi
    else
        tex_packages=("texlive-xetex" "texlive-fonts-recommended" "texlive-latex-recommended" "texlive-latex-extra" "pandoc")
        for package in "${tex_packages[@]}"; do
            if ! dpkg -l | grep -q "^ii  $package " >/dev/null 2>&1; then
                echo "Error: Missing TeX package: $package"
                missing=$((missing + 1))
            fi
        done
    fi

    # Check for required commands
    for cmd in python3 pip jupyter pandoc; do
        if ! command -v $cmd &> /dev/null; then
            echo "Error: $cmd required"
            missing=$((missing + 1))
        fi
    done

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

# Function to setup notebook conversion environment and dependencies
setup_notebook_conversion() {
    echo "Setting up notebook conversion environment..."
    mkdir -p logs/notebook_conversion
    local log_file="logs/notebook_conversion/setup.log"

    # Log all commands and their output
    {
        echo "=== Starting notebook setup at $(date) ==="
        echo "Installing required packages..."
        pip install --quiet "notebook<7.0.0" nbconvert jupyter_contrib_nbextensions

        echo "Installing nbextensions..."
        # Install the extensions
        jupyter contrib nbextension install --user --quiet
        jupyter nbextension enable --user --quiet

        echo "Setting up jupyter config..."
        mkdir -p ~/.jupyter
        if [ ! -f ~/.jupyter/jupyter_notebook_config.py ]; then
            echo "Initializing jupyter notebook config..."
            jupyter notebook --generate-config
            if [ -f "jupyter_notebook_config.py" ]; then
                cp jupyter_notebook_config.py ~/.jupyter/
            fi
        fi
    } >> "$log_file" 2>&1

    if ! jupyter --version >> "$log_file" 2>&1; then
        echo "Error with jupyter installation - check $log_file for details"
        return 1
    fi

    echo "✓ Setup complete"
    return 0
}

# Function to convert notebooks to PDF
convert_notebooks() {
    echo "Starting notebook conversion..."
    local conversion_errors=0

    # Check if there are any notebooks
    shopt -s nullglob
    notebooks=(notebooks/*.ipynb)
    if [ ${#notebooks[@]} -eq 0 ]; then
        echo "No notebooks found in notebooks directory"
        return 0
    fi

    # Process each notebook
    for notebook in "${notebooks[@]}"; do
        basename=$(basename "$notebook" .ipynb)
        local log_file="logs/notebook_conversion/${basename}_conversion.log"

        echo "Converting ${basename}..."
        {
            echo "=== Starting conversion of ${basename} at $(date) ==="
            echo "Attempting PDF conversion..."
            jupyter nbconvert --to pdf "${notebook}" --output-dir="notebooks" || {
                echo "PDF conversion failed, trying HTML path..."
                if jupyter nbconvert --to html "${notebook}" --template basic --output-dir="notebooks" && \
                   pandoc "notebooks/${basename}.html" -o "notebooks/${basename}.pdf" --pdf-engine=xelatex -V geometry:margin=1in; then
                    echo "HTML->PDF conversion successful"
                    rm -f "notebooks/${basename}.html"
                else
                    echo "All conversion attempts failed"
                    conversion_errors=$((conversion_errors + 1))
                fi
            }
        } >> "$log_file" 2>&1

        if [ -f "notebooks/${basename}.pdf" ]; then
            echo "✓ Created ${basename}.pdf"
        else
            echo "⚠ Failed to convert ${basename} - check $log_file for details"
        fi
    done

    if [ $conversion_errors -eq 0 ]; then
        echo "✓ All notebook conversions completed successfully"
    else
        echo "⚠ ${conversion_errors} notebook conversion(s) failed - check logs/notebook_conversion/ for details"
    fi

    return $conversion_errors
}

# Check dependencies
check_dependencies
if [ $? -ne 0 ]; then
    echo "Please install missing dependencies before continuing."
    exit 1
fi

# Create required directories
echo "Creating project directories..."
mkdir -p data
mkdir -p processed_data
mkdir -p models
mkdir -p figures/{exploration,feature_analysis,final_analysis,models,ablation_studies}
mkdir -p analysis_results
mkdir -p logs
mkdir -p notebooks
mkdir -p reports
echo "✓ Project directories created"

# Check directory permissions
for dir in data processed_data models figures analysis_results logs notebooks reports; do
    check_permissions $dir
done

# Set up Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Install dependencies
echo "Installing dependencies..."
pip install --quiet --upgrade pip wheel setuptools > logs/pip_install.log 2>&1
pip install --quiet pyzmq==25.1.2 >> logs/pip_install.log 2>&1
pip install --quiet -r requirements.txt >> logs/pip_install.log 2>&1
echo "✓ Dependencies installed"

echo -e "\nExecuting pipeline stages:"

# Run data preprocessing pipeline
echo "1. Data preprocessing..."
python3 src/main.py > logs/data_preprocessing.log 2>&1
check_file "processed_data/final_processed_data.csv"
echo "✓ Data preprocessing complete"

# Run feature engineering
echo "2. Feature engineering..."
python3 src/data/feature_engineering.py > logs/feature_engineering.log 2>&1
check_file "processed_data/engineered_features.csv"
echo "✓ Feature engineering complete"

# Run initial model training
echo "3. Model training..."
python3 run.py > logs/model_training.log 2>&1
check_file "models/training_results.yaml"
echo "✓ Model training complete"

# Run model evaluation
echo "4. Model evaluation..."
python3 src/models/evaluate.py > logs/model_evaluation.log 2>&1
echo "✓ Model evaluation complete"

# Run ablation studies
echo "5. Ablation studies..."
python3 run_ablation.py > logs/ablation_studies.log 2>&1
check_file "figures/ablation_studies/ablation_results.yaml"
echo "✓ Ablation studies complete"

# Run results analysis
echo "6. Results analysis..."
python3 analyze_results.py > logs/results_analysis.log 2>&1
check_file "analysis_results/analysis_report.md"
echo "✓ Results analysis complete"

# Run model refinement
echo "7. Model refinement..."
python3 refine_models.py > logs/model_refinement.log 2>&1
check_file "models/refined_results.yaml"
echo "✓ Model refinement complete"

# Create visualizations
echo "8. Creating visualizations..."
python3 create_visualizations.py > logs/visualizations.log 2>&1
check_file "figures/final_analysis/model_comparison.png"
echo "✓ Visualizations created"

# Generate final report and convert notebooks
echo "9. Converting notebooks..."
mkdir -p logs/notebook_conversion

# Attempt conversion
setup_notebook_conversion && {
    convert_notebooks
    echo "✓ Notebook processing complete"
} || {
    echo "⚠ Notebook conversion encountered errors - check logs for details"
}

# Print summary
echo -e "\n========================================="
echo "Project execution completed successfully!"
echo "========================================="
echo "Generated files:"
echo "- Processed data: ./processed_data/"
echo "- Model outputs: ./models/"
echo "- Figures: ./figures/"
echo "- Analysis results: ./analysis_results/"
echo "- Final visualizations: ./figures/final_analysis/"
echo "- Logs: ./logs/"
echo "- Notebooks (PDF): ./notebooks/"

# Record completion
echo "Completed at: $(date)" >> logs/execution_history.log

exit 0
# Renewable Energy Adoption Rate Prediction

A machine learning project for predicting and analyzing renewable energy adoption rates across countries.

## Overview

This project implements a comprehensive machine learning pipeline to predict renewable energy adoption patterns using historical data, economic
indicators, and weather patterns. It includes data preprocessing, feature engineering, model training, and evaluation components.

## Project Structure

```
.
├── data/                      # Raw data storage
│   ├── Global Energy Consumption & Renewable Generation/
│   ├── Renewable Energy World Wide 1965-2022/
│   ├── energy_dataset_.csv
│   ├── renewable_energy_and_weather_conditions.csv
│   └── us_renewable_energy_consumption.csv
├── processed_data/            # Processed and engineered data
├── models/                    # Trained models and results
├── figures/                   # Generated visualizations
│   ├── ablation_studies/     # Ablation study results
│   ├── exploration/          # Data exploration plots
│   ├── feature_analysis/     # Feature analysis plots
│   ├── final_analysis/       # Final results visualization
│   └── models/               # Model performance plots
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_feature_analysis.ipynb
├── src/                      # Source code
│   ├── data/                # Data processing modules
│   └── models/              # Model implementations
├── logs/                    # Execution logs
└── requirements.txt         # Project dependencies
```

## Prerequisites

- Python 3.10 or higher
- pip package manager
- LaTeX installation (for PDF report generation)
- Git

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/k-g-j/cs6140-course-project.git
   cd cs6140-course-project
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Option 1: Using the Shell Script (Recommended)

Execute the complete pipeline using the provided shell script:

```bash
chmod +x run.sh
./run.sh
```

This will:

- Set up required directories
- Install dependencies
- Run data preprocessing
- Execute feature engineering
- Train and evaluate models
- Generate visualizations and reports

### Option 2: Step-by-Step Execution

1. Data Preprocessing:
   ```bash
   python src/main.py
   ```

2. Feature Engineering:
   ```bash
   python src/data/feature_engineering.py
   ```

3. Model Training:
   ```bash
   python run.py
   ```

4. Model Evaluation:
   ```bash
   python src/models/evaluate.py
   ```

5. Ablation Studies:
   ```bash
   python run_ablation.py
   ```

6. Results Analysis:
   ```bash
   python analyze_results.py
   ```

7. Generate Visualizations:
   ```bash
   python create_visualizations.py
   ```

## Output Files

After execution, you'll find:

- Processed data in `processed_data/`
- Trained model results in `models/`
- Visualizations in `figures/`
- Analysis reports in `analysis_results/`
- Execution logs in `logs/`

## Jupyter Notebooks

For interactive data exploration and analysis:

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Navigate to the `notebooks/` directory
3. Open:
    - `01_data_exploration.ipynb` for data analysis
    - `02_feature_analysis.ipynb` for feature engineering insights

## Configuration

The project behavior can be customized through:

- `config.yaml`: Main configuration file
- `config_refined.yaml`: Refined model parameters

## Troubleshooting

Common issues and solutions:

1. Missing data files:
    - Ensure all required datasets are in the `data/` directory
    - Check file permissions

2. Memory errors:
    - Increase available RAM
    - Reduce batch sizes in configuration

3. LaTeX errors:
    - Ensure LaTeX is properly installed
    - Check LaTeX logs in the output directory

## License

MIT License - see LICENSE file for details.

## Author

Kate Johnson - CS6140 Machine Learning Course Project
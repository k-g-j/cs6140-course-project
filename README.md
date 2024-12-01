# Renewable Energy Adoption Rate Prediction

A machine learning project for predicting and analyzing renewable energy adoption rates across countries using historical data, economic indicators,
and weather patterns.

## Overview

This project implements a comprehensive machine learning pipeline to predict renewable energy adoption patterns. The system leverages multiple data
sources and employs various machine learning techniques, from traditional statistical methods to advanced deep learning approaches. Our ensemble model
achieved an R² score of 0.6964 (153% improvement over baseline) while reducing computational overhead by 45%.

Key features:

- Advanced feature engineering with temporal and weather patterns
- Multiple model implementations (baseline, advanced, and ensemble)
- Comprehensive ablation studies and model refinement
- Detailed visualizations and analysis reports

## Project Structure

```
.
├── analysis_results/           # Analysis outputs and reports
│   └── analysis_report.md
├── data/                      # Raw data storage
│   ├── Global Energy Consumption & Renewable Generation/
│   ├── Renewable Energy World Wide 1965-2022/
│   ├── energy_dataset_.csv
│   ├── renewable_energy_and_weather_conditions.csv
│   └── us_renewable_energy_consumption.csv
├── figures/                   # Generated visualizations
│   ├── ablation_studies/     # Ablation study results
│   ├── exploration/          # Data exploration plots
│   ├── feature_analysis/     # Feature analysis plots
│   ├── final_analysis/       # Final results visualization
│   └── models/               # Model performance plots
├── final_project_report/     # Final report and diagrams
│   ├── diagrams/            # Generated diagram images
│   ├── group_15_johnson_final_report.md
│   ├── group_15_johnson_final_report.pdf
│   └── mermaid.lua          # Mermaid diagram configuration
├── logs/                     # Execution and error logs
├── models/                   # Trained models and results
├── notebooks/                # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_feature_analysis.ipynb
├── presentation/            # Project presentation files
├── processed_data/         # Processed and engineered data
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   │   ├── feature_engineering.py
│   │   ├── load_data.py
│   │   └── preprocess.py
│   └── models/            # Model implementations
│       ├── ablation_studies.py
│       ├── advanced_models.py
│       ├── baseline_models.py
│       ├── deep_learning_models.py
│       ├── ensemble_models.py
│       ├── evaluate.py
│       └── train.py
└── requirements.txt        # Project dependencies
```

## Prerequisites

- Python 3.10 or higher
- pip package manager
- LaTeX installation (for PDF report generation)
- Git
- At least 8GB RAM recommended

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/k-g-j/cs6140-course-project.git
   cd cs6140-course-project
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install additional LaTeX dependencies (for PDF generation):
    - On Ubuntu/Debian:
      ```bash
      sudo apt-get install texlive-xetex texlive-fonts-recommended texlive-latex-recommended
      ```
    - On macOS:
      ```bash
      brew install mactex
      ```
    - On Windows:
        - Install MiKTeX from https://miktex.org/download

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
- Convert notebooks to PDF
- Generate the final report

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

8. Generate Final Report:
   ```bash
   cd final_project_report
   ./convert_to_pdf.sh
   ```

## Output Files

After execution, you'll find:

- Processed data in `processed_data/`
- Trained model results in `models/`
- Visualizations in `figures/`
- Analysis reports in `analysis_results/`
- Execution logs in `logs/`
- Final report in `final_project_report/group_15_johnson_final_report.pdf`

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
- `config_refined.yaml`: Refined model parameters after ablation studies
- `jupyter_notebook_config.py`: Jupyter notebook settings

## Troubleshooting

Common issues and solutions:

1. Missing data files:
    - Ensure all required datasets are in the `data/` directory
    - Check file permissions
    - Verify CSV file encodings are UTF-8

2. Memory errors:
    - Increase available RAM (8GB minimum recommended)
    - Reduce batch sizes in configuration
    - Check for memory leaks in preprocessing

3. LaTeX errors:
    - Ensure LaTeX is properly installed
    - Check LaTeX logs in the output directory
    - Verify all required LaTeX packages are installed

4. Notebook conversion errors:
    - Check logs in `logs/notebook_conversion/`
    - Verify Jupyter installation
    - Install missing Jupyter extensions

## Project Components

### Data Processing

- Feature engineering with temporal patterns
- Weather data integration
- Economic indicator processing

### Models Implemented

- Baseline: Linear Regression, Ridge, Lasso, ARIMA
- Advanced: Random Forest, Gradient Boosting, SVR
- Deep Learning: LSTM, CNN
- Ensemble Methods: Stacking, Voting

### Analysis Tools

- Comprehensive ablation studies
- Feature importance analysis
- Model performance visualization
- Time series analysis

### Key Results

- Best performing model (Ensemble): R² = 0.6964
- RMSE reduction: 31%
- MAE improvement: 35%
- Computational efficiency: 45% reduction in memory usage

## License

MIT License - see LICENSE file for details.

## Authors

Kate Johnson - CS6140 Machine Learning Course Project

## References

Key papers and resources:

1. Mahjoub, S., et al. (2022). "Predicting Energy Consumption Using LSTM, Multi-Layer GRU and Drop-GRU Neural Networks"
2. Kaur, D., et al. (2022). "Energy forecasting in smart grid systems: recent advancements in probabilistic deep learning"
3. Albuquerque, P. C., et al. (2022). "Machine learning models for forecasting power electricity consumption using a high dimensional dataset"
4. Mathumitha, R., et al. (2024). "Intelligent deep learning techniques for energy consumption forecasting in smart buildings: a review"
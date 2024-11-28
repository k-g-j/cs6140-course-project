"""Script for analyzing model results and generating insights."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """Class to analyze model results and generate reports."""

    def __init__(self, results_path: Path, config_path: Path):
        self.results_path = results_path
        self.config_path = config_path
        self.output_dir = Path('analysis_results')
        self.output_dir.mkdir(exist_ok=True)

        # Load results and config
        if not self.results_path.exists():
            logger.warning(f"Results file not found at {self.results_path}")
            self.results = {}
        else:
            with open(self.results_path) as f:
                self.results = yaml.safe_load(f)

        if not self.config_path.exists():
            logger.error(f"Config file not found at {self.config_path}")
            self.config = {}
        else:
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)

    def analyze_model_performance(self):
        """Analyze performance metrics across models."""
        # Extract performance metrics
        metrics = {}
        for model_type in ['baseline', 'advanced']:
            for model_name, model_results in self.results.get(model_type, {}).items():
                metrics[f"{model_type}_{model_name}"] = {
                    'r2': model_results.get('r2', 0),
                    'rmse': model_results.get('rmse', 0),
                    'mae': model_results.get('mae', 0)
                }

        if not metrics:
            logger.warning("No model performance metrics found.")
            return pd.DataFrame(), {}

        # Create performance comparison DataFrame
        df = pd.DataFrame(metrics).T

        # Generate insights
        insights = {
            'best_model': df['r2'].idxmax(),
            'best_r2': df['r2'].max(),
            'worst_model': df['r2'].idxmin(),
            'worst_r2': df['r2'].min(),
            'avg_r2': df['r2'].mean()
        }

        return df, insights

    def analyze_feature_importance(self):
        """Analyze feature importance across models."""
        importance_data = {}
        for model_type in ['baseline', 'advanced']:
            for model_name, model_results in self.results.get(model_type, {}).items():
                if 'feature_importances' in model_results:
                    importance_data[f"{model_type}_{model_name}"] = model_results[
                        'feature_importances']

        # Convert to DataFrame
        if importance_data:
            df = pd.DataFrame(importance_data)
            return df
        logger.info("No feature importance data found.")
        return None

    def analyze_ablation_results(self):
        """Analyze ablation study results."""
        ablation_path = Path('figures/ablation_studies/ablation_results.yaml')
        if not ablation_path.exists():
            logger.warning("No ablation results found.")
            return None

        with open(ablation_path) as f:
            ablation_results = yaml.safe_load(f)

        if not ablation_results:
            logger.warning("Ablation results file is empty.")
            return None

        # Analyze impact of different components
        impacts = {}
        for study_type, results in ablation_results.items():
            if not results:
                logger.warning(f"No results found for study type '{study_type}'.")
                continue

            # Depending on the structure of results, adjust the analysis
            if study_type == 'feature_importance':
                impacts[study_type] = {
                    'overall_impact': sum(res['relative_impact'] for res in results.values()),
                    'average_impact': np.mean([res['relative_impact'] for res in results.values()]),
                    'most_impacted':
                        max(results.items(), key=lambda x: abs(x[1]['relative_impact']))[0]
                }
            elif study_type == 'model_complexity':
                # Example analysis: Identify the configuration with the highest R²
                best_config = max(results.items(), key=lambda x: x[1]['r2_score'])
                impacts[study_type] = {
                    'best_configuration': best_config[0],
                    'best_r2': best_config[1]['r2_score']
                }
            elif study_type == 'data_volume':
                # Example analysis: Trend of R² with increasing data size
                sizes = sorted(results.keys(), key=lambda x: int(x.split('_')[1]))
                r2_scores = [results[size]['r2_score'] for size in sizes]
                impacts[study_type] = {
                    'sizes': sizes,
                    'r2_scores': r2_scores
                }

        return impacts

    def generate_report(self):
        """Generate comprehensive analysis report."""
        performance_df, performance_insights = self.analyze_model_performance()
        feature_importance_df = self.analyze_feature_importance()
        ablation_impacts = self.analyze_ablation_results()

        report = ["# Model Analysis Report", "\n"]

        # Performance Summary
        if not performance_df.empty:
            report.append("## 1. Model Performance Summary\n")
            report.append(
                f"- **Best performing model:** {performance_insights['best_model']} (R² = {performance_insights['best_r2']:.4f})")
            report.append(f"- **Average R² across models:** {performance_insights['avg_r2']:.4f}\n")
            report.append("### Detailed Performance Metrics:\n")
            report.append(performance_df.to_markdown())
        else:
            report.append("## 1. Model Performance Summary\n")
            report.append("No performance metrics available.\n")

        # Feature Importance
        if feature_importance_df is not None:
            report.append("\n## 2. Feature Importance Analysis\n")
            report.append(feature_importance_df.to_markdown())
        else:
            report.append("\n## 2. Feature Importance Analysis\n")
            report.append("No feature importance data available.\n")

        # Ablation Studies Insights
        if ablation_impacts:
            report.append("\n## 3. Ablation Study Insights\n")
            for study_type, impact in ablation_impacts.items():
                report.append(f"\n### {study_type.replace('_', ' ').title()}\n")
                if study_type == 'feature_importance':
                    report.append(f"- **Overall Impact:** {impact['overall_impact']:.2f}%")
                    report.append(f"- **Average Impact:** {impact['average_impact']:.2f}%")
                    report.append(f"- **Most Impacted Feature Group:** {impact['most_impacted']}\n")
                elif study_type == 'model_complexity':
                    report.append(f"- **Best Configuration:** {impact['best_configuration']}")
                    report.append(f"- **Best R² Score:** {impact['best_r2']:.4f}\n")
                elif study_type == 'data_volume':
                    sizes = ', '.join(impact['sizes'])
                    r2_scores = ', '.join([f"{score:.4f}" for score in impact['r2_scores']])
                    report.append(f"- **Sizes:** {sizes}")
                    report.append(f"- **R² Scores:** {r2_scores}\n")
        else:
            report.append("\n## 3. Ablation Study Insights\n")
            report.append("No ablation study results available.\n")

        # Save report
        report_path = self.output_dir / 'analysis_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))

        logger.info(f"Analysis report saved to {report_path}")
        return report_path


def main():
    """Main execution function."""
    try:
        results_path = Path('models/training_results.yaml')
        config_path = Path('config.yaml')

        analyzer = ResultAnalyzer(results_path, config_path)
        report_path = analyzer.generate_report()

        logger.info(f"Analysis completed successfully. Report saved to {report_path}")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

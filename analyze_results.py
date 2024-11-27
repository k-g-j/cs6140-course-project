"""Script for analyzing model results and generating insights."""
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResultAnalyzer:
    def __init__(self, results_path: Path, config_path: Path):
        self.results_path = results_path
        self.config_path = config_path
        self.output_dir = Path('analysis_results')
        self.output_dir.mkdir(exist_ok=True)

        # Load results and config
        with open(results_path) as f:
            self.results = yaml.safe_load(f)
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def analyze_model_performance(self):
        """Analyze performance metrics across models."""
        # Extract performance metrics
        metrics = {}
        for model_type in ['baseline', 'advanced']:
            for model_name, model_results in self.results[model_type].items():
                metrics[f"{model_type}_{model_name}"] = {
                    'r2': model_results.get('r2', 0),
                    'rmse': model_results.get('rmse', 0),
                    'mae': model_results.get('mae', 0)
                }

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
            for model_name, model_results in self.results[model_type].items():
                if 'feature_importances' in model_results:
                    importance_data[f"{model_type}_{model_name}"] = model_results[
                        'feature_importances']

        # Convert to DataFrame
        if importance_data:
            df = pd.DataFrame(importance_data)
            return df
        return None

    def analyze_ablation_results(self):
        """Analyze ablation study results."""
        ablation_path = self.output_dir.parent / 'ablation_results.yaml'
        if not ablation_path.exists():
            logger.warning("No ablation results found")
            return None

        with open(ablation_path) as f:
            ablation_results = yaml.safe_load(f)

        # Analyze impact of different components
        impacts = {}
        for study_type, results in ablation_results.items():
            impacts[study_type] = {
                'max_impact': max(r['impact'] for r in results),
                'avg_impact': sum(r['impact'] for r in results) / len(results),
                'most_important': max(results, key=lambda x: x['impact'])['component']
            }

        return impacts

    def generate_report(self):
        """Generate comprehensive analysis report."""
        performance_df, performance_insights = self.analyze_model_performance()
        feature_importance_df = self.analyze_feature_importance()
        ablation_impacts = self.analyze_ablation_results()

        report = ["# Model Analysis Report\n"]

        # Performance Summary
        report.append("## Model Performance Summary\n")
        report.append(
            f"- Best performing model: {performance_insights['best_model']} (R² = {performance_insights['best_r2']:.4f})")
        report.append(f"- Average model performance: R² = {performance_insights['avg_r2']:.4f}")
        report.append(f"\nDetailed Performance Metrics:\n")
        report.append(performance_df.to_string())

        # Feature Importance
        if feature_importance_df is not None:
            report.append("\n\n## Feature Importance Analysis\n")
            report.append(feature_importance_df.to_string())

        # Ablation Results
        if ablation_impacts:
            report.append("\n\n## Ablation Study Insights\n")
            for study_type, impact in ablation_impacts.items():
                report.append(f"\n### {study_type}")
                report.append(f"- Maximum impact: {impact['max_impact']:.2f}")
                report.append(f"- Average impact: {impact['avg_impact']:.2f}")
                report.append(f"- Most important component: {impact['most_important']}")

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
        raise


if __name__ == "__main__":
    main()

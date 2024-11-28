"""Implementation of ablation studies for renewable energy prediction models."""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.train import ModelTrainer

logger = logging.getLogger(__name__)


class AblationStudy:
    """Class to perform ablation studies on trained models."""

    def __init__(self, config_path: str):
        """Initialize ablation study with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.output_dir = Path('figures/ablation_studies')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.trainer = ModelTrainer(config_path)
        self.results = {}

        # Initialize preprocessors
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess data by handling missing values and scaling."""
        # Handle missing values
        X_processed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        y_processed = y.copy()

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_processed),
            columns=X_processed.columns
        )

        return X_scaled, y_processed

    def feature_importance_ablation(self,
                                    X: pd.DataFrame,
                                    y: pd.Series,
                                    feature_groups: Optional[Dict[str, List[str]]] = None) -> Dict:
        """Study impact of different feature groups."""
        # Preprocess the full dataset first
        X_processed, y_processed = self.preprocess_data(X, y)

        if feature_groups is None:
            feature_groups = {
                'temporal': [col for col in X_processed.columns if
                             'lag' in col or 'rolling' in col],
                'generation': [col for col in X_processed.columns if 'generation' in col],
                'weather': [col for col in X_processed.columns if
                            any(x in col for x in ['temp', 'wind', 'solar'])],
                'economic': [col for col in X_processed.columns if
                             any(x in col for x in ['gdp', 'consumption', 'share'])]
            }

        results = {}
        baseline_score = None

        # Train model with all features first
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2,
                                                            random_state=42)
        self.trainer.train_models(X_train, X_test, y_train, y_test)
        baseline_score = self.trainer.advanced_models.metrics['gradient_boosting']['r2']

        # Test removing each feature group
        for group_name, features in feature_groups.items():
            remaining_features = [col for col in X_processed.columns if col not in features]
            if remaining_features:
                X_subset = X_processed[remaining_features]
                X_train, X_test, y_train, y_test = train_test_split(X_subset, y_processed,
                                                                    test_size=0.2, random_state=42)

                self.trainer.train_models(X_train, X_test, y_train, y_test)
                score = self.trainer.advanced_models.metrics['gradient_boosting']['r2']

                impact = baseline_score - score
                results[group_name] = {
                    'removed_features': features,
                    'performance_impact': impact,
                    'relative_impact': (impact / baseline_score) * 100 if baseline_score != 0 else 0
                }

        self.results['feature_importance'] = results
        return results

    def model_complexity_ablation(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Study impact of model complexity."""
        # Preprocess data first
        X_processed, y_processed = self.preprocess_data(X, y)

        results = {}
        n_estimators_range = [50, 100, 200, 300]
        max_depth_range = [5, 10, 15, None]

        for n_estimators in n_estimators_range:
            for max_depth in max_depth_range:
                model_config = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth
                }

                X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed,
                                                                    test_size=0.2, random_state=42)
                # Update model configuration
                self.trainer.advanced_models.config = {'rf_n_estimators': n_estimators,
                                                       'rf_max_depth': max_depth}
                self.trainer.train_models(X_train, X_test, y_train, y_test)

                score = self.trainer.advanced_models.metrics['random_forest']['r2']
                results[f'rf_n{n_estimators}_d{max_depth}'] = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'r2_score': score
                }

        self.results['model_complexity'] = results
        return results

    def data_volume_ablation(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Study impact of training data volume."""
        # Preprocess data first
        X_processed, y_processed = self.preprocess_data(X, y)

        results = {}
        train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

        for size in train_sizes:
            if size < 1.0:
                X_subset, _, y_subset, _ = train_test_split(X_processed, y_processed,
                                                            train_size=size, random_state=42)
            else:
                X_subset, y_subset = X_processed, y_processed

            X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2,
                                                                random_state=42)
            self.trainer.train_models(X_train, X_test, y_train, y_test)

            results[f'size_{int(size * 100)}'] = {
                'training_size': size,
                'n_samples': len(X_train),
                'r2_score': self.trainer.advanced_models.metrics['gradient_boosting']['r2']
            }

        self.results['data_volume'] = results
        return results

    def visualize_results(self):
        """Create visualizations for ablation study results."""
        # Feature importance visualization
        if 'feature_importance' in self.results:
            plt.figure(figsize=(12, 6))
            feature_impacts = [res['relative_impact'] for res in
                               self.results['feature_importance'].values()]
            feature_names = list(self.results['feature_importance'].keys())

            sns.barplot(x=feature_names, y=feature_impacts)
            plt.title('Feature Group Importance (% Impact on Performance)')
            plt.xticks(rotation=45)
            plt.ylabel('Relative Impact (%)')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_importance_ablation.png', dpi=300)
            plt.close()

        # Model complexity visualization
        if 'model_complexity' in self.results:
            complexity_data = pd.DataFrame(self.results['model_complexity']).T
            plt.figure(figsize=(12, 6))

            pivot_data = complexity_data.pivot(
                index='n_estimators',
                columns='max_depth',
                values='r2_score'
            )

            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis')
            plt.title('Model Performance vs Complexity (Random Forest)')
            plt.xlabel('Max Depth')
            plt.ylabel('Number of Estimators')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'model_complexity_ablation.png', dpi=300)
            plt.close()

        # Data volume visualization
        if 'data_volume' in self.results:
            plt.figure(figsize=(10, 6))
            volumes = [res['training_size'] * 100 for res in self.results['data_volume'].values()]
            scores = [res['r2_score'] for res in self.results['data_volume'].values()]

            plt.plot(volumes, scores, marker='o')
            plt.xlabel('Training Data Size (%)')
            plt.ylabel('R² Score')
            plt.title('Learning Curve - Performance vs Data Volume')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'data_volume_ablation.png', dpi=300)
            plt.close()

    def generate_report(self) -> str:
        """Generate a detailed report of ablation study findings."""
        report = ["# Ablation Studies Report", "=" * 50, ""]

        # Feature importance findings
        if 'feature_importance' in self.results:
            report.extend([
                "\n## 1. Feature Importance Analysis",
                "-" * 30
            ])

            sorted_features = sorted(
                self.results['feature_importance'].items(),
                key=lambda x: abs(x[1]['relative_impact']),
                reverse=True
            )

            for feature, impact in sorted_features:
                report.append(f"\n### {feature}")
                report.append(f"- **Performance Impact:** {impact['relative_impact']:.2f}%")
                report.append(f"- **Removed Features:** {', '.join(impact['removed_features'])}")

        # Model complexity findings
        if 'model_complexity' in self.results:
            report.extend([
                "\n## 2. Model Complexity Analysis",
                "-" * 30
            ])

            best_config = max(
                self.results['model_complexity'].items(),
                key=lambda x: x[1]['r2_score']
            )

            report.append(f"\n### Best Configuration:")
            report.append(f"- **Number of Estimators:** {best_config[1]['n_estimators']}")
            report.append(f"- **Max Depth:** {best_config[1]['max_depth']}")
            report.append(f"- **R² Score:** {best_config[1]['r2_score']:.4f}")

        # Data volume findings
        if 'data_volume' in self.results:
            report.extend([
                "\n## 3. Data Volume Analysis",
                "-" * 30
            ])

            volume_results = self.results['data_volume']
            max_score = max(res['r2_score'] for res in volume_results.values())
            min_score = min(res['r2_score'] for res in volume_results.values())

            report.append(f"\n- **Maximum R² (100% data):** {max_score:.4f}")
            report.append(f"- **Minimum R² (20% data):** {min_score:.4f}")
            if min_score != 0:
                performance_gain = ((max_score - min_score) / min_score) * 100
            else:
                performance_gain = 0
            report.append(f"- **Performance Gain:** {performance_gain:.1f}%")

        # Save report
        report_path = self.output_dir / 'ablation_study_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))

        return '\n'.join(report)

    def run_all_studies(self, X: pd.DataFrame, y: pd.Series):
        """Run all ablation studies and save results."""
        try:
            logger.info("Starting ablation studies...")

            # Create results dictionary
            results = {
                'feature_importance': {},
                'model_complexity': {},
                'data_volume': {}
            }

            # Run feature importance ablation
            logger.info("Running feature importance ablation...")
            results['feature_importance'] = self.feature_importance_ablation(X, y)

            # Run model complexity ablation
            logger.info("Running model complexity ablation...")
            results['model_complexity'] = self.model_complexity_ablation(X, y)

            # Run data volume ablation
            logger.info("Running data volume ablation...")
            results['data_volume'] = self.data_volume_ablation(X, y)

            # Save results
            results_path = self.output_dir / 'ablation_results.yaml'
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Convert numpy types to native Python types
            def convert_to_native(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_native(x) for x in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            serializable_results = convert_to_native(results)

            with open(results_path, 'w') as f:
                yaml.dump(serializable_results, f, default_flow_style=False)

            # Create visualizations
            logger.info("Generating visualizations...")
            self.visualize_results()

            # Generate report
            logger.info("Generating ablation study report...")
            report = self.generate_report()

            # Save report
            report_path = self.output_dir / 'ablation_study_report.md'
            with open(report_path, 'w') as f:
                f.write(report)

            logger.info("Ablation studies completed successfully!")
            return results

        except Exception as e:
            logger.error(f"Error in ablation studies: {str(e)}")
            raise


def main():
    """Main execution function."""
    try:
        # Load your data here
        config_path = 'config.yaml'
        ablation = AblationStudy(config_path)

        # Load processed data
        data_path = Path('processed_data/final_processed_data.csv')
        if not data_path.exists():
            raise FileNotFoundError(f"Processed data not found at {data_path}")

        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)

        # Prepare features and target
        target_col = ablation.config['training']['target_column']
        feature_cols = ablation.config['training']['feature_columns']

        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in data.")
            raise ValueError(f"Target column '{target_col}' not found in data.")

        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            logger.error(f"Missing feature columns in data: {missing_features}")
            raise ValueError(f"Missing feature columns in data: {missing_features}")

        X = df[feature_cols]
        y = df[target_col]

        # Run ablation studies
        ablation.run_all_studies(X, y)

    except Exception as e:
        logger.error(f"Ablation studies failed: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup code
        logger.info("Execution completed")


if __name__ == "__main__":
    main()

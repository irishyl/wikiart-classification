import os
import json
import torch
import joblib
import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class BaselineModelTrainer:
    def __init__(self, model_type: str = "svm", output_dir: str = "output", shuffle_data: bool = False, **model_params):
        """
        Initialize the baseline model trainer.

        Parameters
        ----------
        model_type : str
            "svm" or "rf"
        output_dir : str
            Base directory where results will be saved.
        shuffle_data : bool
            Whether to shuffle the training data.
        **model_params : dict
            Additional params for model initialization.
        """
        self.model_type = model_type.lower()
        self.model_params = model_params
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.scaler = None
        self.shuffle_data = shuffle_data

        # Create a model-specific directory
        self.base_output_dir = os.path.join(output_dir, self.model_type)
        os.makedirs(self.base_output_dir, exist_ok=True)

        # Create a timestamped directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.base_output_dir, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)

    def _load_data(self, path: str):
        data = torch.load(path)
        X = data['features'].numpy()
        y = data['labels'].numpy()
        return X, y

    def load_datasets(self, train_path: str, valid_path: str):
        print(f"Loading training data from {train_path}...")
        self.X_train, self.y_train = self._load_data(train_path)
        print(f"Training data loaded: {self.X_train.shape[0]} samples.")

        print(f"Loading validation data from {valid_path}...")
        self.X_val, self.y_val = self._load_data(valid_path)
        print(f"Validation data loaded: {self.X_val.shape[0]} samples.")

        if self.shuffle_data:
            # Shuffle training data
            train_perm = np.random.permutation(self.X_train.shape[0])
            self.X_train = self.X_train[train_perm]
            self.y_train = self.y_train[train_perm]
            print("Training data shuffled.")

    def initialize_model(self):
        if self.model_type == "svm":
            kernel = self.model_params.pop('kernel', 'rbf')
            C = self.model_params.pop('C', 1.0)
            print(f"Initializing SVM (kernel={kernel}, C={C}, params={self.model_params})")
            self.model = SVC(kernel=kernel, C=C, **self.model_params)
        elif self.model_type == "rf":
            n_estimators = self.model_params.pop('n_estimators', 100)
            print(f"Initializing RF (n_estimators={n_estimators}, params={self.model_params})")
            self.model = RandomForestClassifier(n_estimators=n_estimators, **self.model_params)
        else:
            raise ValueError(f"Unsupported model_type '{self.model_type}'. Choose 'svm' or 'rf'.")

    def preprocess_data(self):
        print("Preprocessing data (scaling features)...")
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)

    def tune_hyperparameters(self, param_grid, search_type="grid", cv=5, n_jobs=-1, scoring="accuracy"):
        """
        Tune hyperparameters using grid search or randomized search.
        """
        if self.model_type == "svm":
            base_model = SVC()
        elif self.model_type == "rf":
            base_model = RandomForestClassifier()
        else:
            raise ValueError("Unsupported model type for tuning.")

        if search_type == "grid":
            search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=cv, n_jobs=n_jobs,
                                  scoring=scoring, verbose=2)
        else:
            search = RandomizedSearchCV(estimator=base_model, param_distributions=param_grid, cv=cv, n_jobs=n_jobs,
                                        scoring=scoring, n_iter=20, verbose=2)

        print("Starting hyperparameter tuning...")
        search.fit(self.X_train, self.y_train)

        self.model = search.best_estimator_
        print(f"Best parameters found: {search.best_params_}")
        print(f"Best CV accuracy: {search.best_score_:.4f}")

        # Save the tuning results
        cv_results = pd.DataFrame(search.cv_results_)
        cv_results_path = os.path.join(self.run_dir, "tuning_results.csv")
        cv_results.to_csv(cv_results_path, index=False)
        print(f"Tuning results saved to {cv_results_path}")

        # Save the best model
        best_model_path = os.path.join(self.run_dir, "best_model.pkl")
        joblib.dump(self.model, best_model_path)
        print(f"Best model saved to {best_model_path}")

    def fit(self):
        if self.model is None:
            self.initialize_model()

        print("Starting model training...")
        self.model.fit(self.X_train, self.y_train)
        print("Training complete.")

    def evaluate(self, save_results: bool = True):
        print("Evaluating on validation set...")
        preds = self.model.predict(self.X_val)

        # Basic accuracy
        acc = accuracy_score(self.y_val, preds)
        print(f"Validation Accuracy: {acc:.4f}")

        # Classification report (precision, recall, f1-score)
        class_report_dict = classification_report(self.y_val, preds, output_dict=True)
        class_report_formatted = self._format_classification_report(class_report_dict)
        print("\nClassification Report:")
        print(class_report_formatted)

        # Confusion matrix
        cm = confusion_matrix(self.y_val, preds)
        cm_table = self._format_confusion_matrix(cm)
        print("\nConfusion Matrix:")
        print(cm_table)

        results = {
            "model_type": self.model_type,
            "params": self.model_params,
            "validation_accuracy": acc,
            "classification_report": class_report_dict,
            "confusion_matrix": cm.tolist()
        }

        if save_results:
            results_path = os.path.join(self.run_dir, "model_results.json")
            # Always create a new results file in the timestamped run directory
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {results_path}")

        return results

    def _format_classification_report(self, report_dict):
        headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        table = []
        for key, values in report_dict.items():
            if isinstance(values, dict):
                table.append([
                    key,
                    f"{values.get('precision',0):.3f}",
                    f"{values.get('recall',0):.3f}",
                    f"{values.get('f1-score',0):.3f}",
                    int(values.get('support',0))
                ])
            else:
                # Accuracy is a single float, let's display it in a row
                if key == 'accuracy':
                    table.append([
                        key,
                        "-", # no precision
                        "-", # no recall
                        f"{values:.3f}",
                        "-"  # no support
                    ])

        return tabulate(table, headers, tablefmt="pretty")

    def _format_confusion_matrix(self, cm):
        headers = ["Pred\\True"] + [str(i) for i in range(cm.shape[1])]
        table = []
        for i, row in enumerate(cm):
            table.append([str(i)] + list(map(str, row)))
        return tabulate(table, headers, tablefmt="pretty")

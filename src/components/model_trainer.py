import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# ================= CONFIG =================
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


# ================= MODEL TRAINER =================
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            # Split input and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # ================= MODELS =================
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
            }

            # ================= HYPERPARAMETERS =================
            params = {
                "Linear Regression": {},

                "Decision Tree": {
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                },

                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10],
                },

                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5],
                },

                "AdaBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1],
                },
            }

            logging.info("Starting hyperparameter tuning using GridSearchCV")

            model_report = {}

            # ================= GRID SEARCH =================
            for model_name in models:
                model = models[model_name]
                param = params[model_name]

                if param:
                    gs = GridSearchCV(model, param, cv=3, n_jobs=-1)
                    gs.fit(X_train, y_train)
                    best_model = gs.best_estimator_
                else:
                    # For Linear Regression (no hyperparameters)
                    best_model = model.fit(X_train, y_train)

                # Predict on test data
                y_test_pred = best_model.predict(X_test)
                score = r2_score(y_test, y_test_pred)

                model_report[model_name] = score

            # ================= BEST MODEL =================
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            # Refit best model on full training data
            if params[best_model_name]:
                gs = GridSearchCV(best_model, params[best_model_name], cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                best_model.fit(X_train, y_train)

            # ================= SAVE MODEL =================
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            logging.info("Best model saved successfully")

            # Final prediction score
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)

# pip install autogluon.tabular[mitra]
import numpy as np
import pandas as pd
import torch

from autogluon.tabular import TabularDataset, TabularPredictor

from utils import load_features


class AutGluonModel:
    def __init__(self):
        # look up things from solvent names
        self.feature_lookup = {name: list(d.values()) for name, d in load_features("spange_descriptors").to_dict('index').items()}
        for name, d in load_features("acs_pca_descriptors").to_dict('index').items():
            self.feature_lookup[name] += list(d.values())

        self.product_2_model = TabularPredictor(
            label="target",
            problem_type="regression",
        )
        self.product_3_model = TabularPredictor(
            label="target",
            problem_type="regression",
        )

    def _get_ds(self, X: pd.DataFrame, target: pd.DataFrame | None) -> TabularDataset:
        if "SolventB%" in X.columns:  # multisolvent
            features = np.concat((
                X[["Residence Time", "Temperature", "SolventB%"]].values,
                np.array([self.feature_lookup[a] for a in X["SOLVENT A NAME"]]),
                np.array([self.feature_lookup[b] for b in X["SOLVENT B NAME"]]),
            ), axis=1)
        else:  # single solvent
            features = np.concat((
                X[["Residence Time", "Temperature"]].values,
                np.array([self.feature_lookup[s] for s in X["SOLVENT NAME"]]),
            ), axis=1)
        df = pd.DataFrame(data=features, columns=[f'feature_{i}' for i in range(features.shape[1])])
        if target is not None:
            df["target"] = target.values
        return TabularDataset(df)

    def train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        ds = self._get_ds(X_train, y_train[["Product 2"]])
        self.product_2_model.fit(
            ds,
            # presets='best',  # 'extreme',
            hyperparameters={
                'MITRA': {'fine_tune': False}  # {'fine_tune': True, 'fine_tune_steps': 10}
            },
            time_limit=60,
        )
        ds = self._get_ds(X_train, y_train[["Product 3"]])
        self.product_3_model.fit(
            ds,
            # presets='best',  # 'extreme',
            hyperparameters={
                'MITRA': {'fine_tune': False}  # {'fine_tune': True, 'fine_tune_steps': 10}
            },
            time_limit=60,
        )

    def predict(self, X: pd.DataFrame) -> torch.Tensor:
        ds = self._get_ds(X, None)
        prod_2_pred: np.ndarray = self.product_2_model.predict(ds, as_pandas=False)
        prod_3_pred: np.ndarray = self.product_3_model.predict(ds, as_pandas=False)
        sm_pred = 1 - (prod_2_pred + prod_3_pred)
        pred = np.stack((prod_2_pred, prod_3_pred, sm_pred), axis=1)
        return torch.from_numpy(pred)

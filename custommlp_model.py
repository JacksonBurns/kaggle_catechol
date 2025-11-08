from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset

from torch import nn

from utils import load_features

class SmallResNetMLP(pl.LightningModule):
    def __init__(self, input_dim, output_dim=1, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        hidden_dim = 32

        # Residual MLP block
        def res_block(dim_in, dim_out):
            return nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.BatchNorm1d(dim_out),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(dim_out, dim_out),
                nn.BatchNorm1d(dim_out),
            )

        self.bn_in = nn.BatchNorm1d(input_dim)
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.block1 = res_block(hidden_dim, hidden_dim)
        self.block2 = res_block(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.bn_in(x)
        x = self.relu(self.fc_in(x))
        # Residual connections
        x = self.relu(x + self.block1(x))
        x = self.relu(x + self.block2(x))
        return self.softmax(self.fc_out(x))
    
    def predict_step(self, batch):
        x = batch[0]
        return self(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class CustomMLPModel:
    def __init__(self):
        self.spange_lookup = {name: list(d.values()) for name, d in load_features("spange_descriptors").to_dict('index').items()}
        self.model = None

    def _get_data(self, X: pd.DataFrame):
        if "SolventB%" in X.columns:  # multisolvent
            perc = X["SolventB%"].to_numpy()
            X = np.concat((
                X[["Residence Time", "Temperature"]].values,
                np.array([self.spange_lookup[a] for a in X["SOLVENT A NAME"]]) * (1 - perc)[:, None],
                np.array([self.spange_lookup[b] for b in X["SOLVENT B NAME"]]) * perc[:, None],
            ), axis=1)
        else:  # single solvent
            X = np.concat((
                X[["Residence Time", "Temperature"]].values,
                np.array([self.spange_lookup[s] for s in X["SOLVENT NAME"]]),
            ), axis=1)
        return torch.from_numpy(np.array(X, dtype=np.float32))

    def train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        pl.seed_everything(42)
        # prepare the data
        y = torch.from_numpy(y_train.values).to(torch.float32)
        X = self._get_data(X_train)

        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # initialize model
        input_dim = X.shape[1]
        self.model = SmallResNetMLP(input_dim=input_dim, output_dim=y.shape[1], lr=1e-3)

        # fit the model
        outdir = Path("mlp_output") / datetime.now().strftime("%Y-%d-%m_%H-%M-%S")
        tensorboard_logger = TensorBoardLogger(
            outdir,
            name="tensorboard_logs",
            default_hp_metric=False,
        )
        callbacks = [
            StochasticWeightAveraging(
                swa_lrs=0.001,
                swa_epoch_start=0.60,
                annealing_epochs=4,
            )
        ]
        trainer = pl.Trainer(
            max_epochs=100,
            gradient_clip_val=1.0,
            logger=tensorboard_logger,
            log_every_n_steps=1,
            enable_checkpointing=False,
            check_val_every_n_epoch=1,
            callbacks=callbacks,
        )
        trainer.fit(self.model, train_loader)

    def predict(self, X: pd.DataFrame) -> torch.Tensor:
        # prepare the data
        X = self._get_data(X)
        dataset = TensorDataset(X)
        inference_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
        )
        return torch.cat(trainer.predict(self.model, dataloaders=inference_loader), dim=0)

# adapted from this publicly posted code:
# https://www.kaggle.com/code/omarafik/system-malfunction-v1
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset, random_split

from torch import nn

from utils import load_features

class SmallResNetMLP(pl.LightningModule):
    def __init__(self, input_dim, output_dim=1, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(self, x):
        return self.log_softmax(self.net(x))
    
    def predict_step(self, batch):
        x = batch[0]
        return torch.exp(self(x))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        log_prob = self(x)
        y = torch.clamp(y, min=1e-8)  # avoid log(0)
        loss = self.criterion(log_prob, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        log_prob = self(x)
        y = torch.clamp(y, min=1e-8)  # avoid log(0)
        loss = self.criterion(log_prob, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}


class CustomMLPModel:
    def __init__(self):
        self.spange_lookup = {name: list(d.values()) for name, d in load_features("spange_descriptors").to_dict('index').items()}
        self.model = None

    def _get_data(self, X: pd.DataFrame):
        if "SolventB%" in X.columns:  # multisolvent
            perc = X["SolventB%"].to_numpy()
            X = np.concat((
                X[["Residence Time", "Temperature"]].values,
                np.array([self.spange_lookup[a] for a in X["SOLVENT A NAME"]]) * (1 - perc)[:, None] + 
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
        train_size = int(X.shape[0] * 0.90)
        g = torch.Generator()
        g.manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, (train_size, X.shape[0] - train_size), g)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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
            EarlyStopping(
                monitor="val_loss",
                patience=10,
            ),
            ModelCheckpoint(
                outdir / "checkpoints",
                monitor="val_loss",
            ),
        ]
        trainer = pl.Trainer(
            max_epochs=100,
            gradient_clip_val=1.0,
            logger=tensorboard_logger,
            log_every_n_steps=1,
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            callbacks=callbacks,
        )
        trainer.fit(self.model, train_loader, val_loader)
        
        ckpt_path = trainer.checkpoint_callback.best_model_path
        self.model = self.model.__class__.load_from_checkpoint(ckpt_path)

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

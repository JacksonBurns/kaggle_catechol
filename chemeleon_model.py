from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from chemprop import featurizers, nn
from chemprop.data import (
    MoleculeDatapoint,
    MoleculeDataset,
    MulticomponentDataset,
    build_dataloader,
)
from chemprop.models import MulticomponentMPNN
from chemprop.nn import RegressionFFN, ScaleTransform
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from utils import load_features


class CheMeleonModel:
    def __init__(self):
        # looking up smiles from solvent names
        self.lookup = load_features("smiles")["solvent smiles"].to_dict()

        # retrieve model checkpoint, if not present
        ckpt_dir = Path().home() / ".chemprop"
        ckpt_dir.mkdir(exist_ok=True)
        self.mp_path = ckpt_dir / "chemeleon_mp.pt"
        if not self.mp_path.exists():
            urlretrieve(
                r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                self.mp_path,
            )

        # needed for stately inference
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.model = None

    def _get_datapoints(self, X: pd.DataFrame, targets: np.ndarray | None):
        if "SolventB%" in X.columns:  # multisolvent
            x_d = X[["Residence Time", "Temperature", "SolventB%"]].values
            smiles = [[self.lookup[a], self.lookup[b]] for a, b in zip(X["SOLVENT A NAME"], X["SOLVENT B NAME"])]
        else:  # single solvent
            x_d = X[["Residence Time", "Temperature"]].values
            smiles = [[self.lookup[name]] for name in X["SOLVENT NAME"]]
        all_data = []
        for i in range(len(smiles)):
            sublist = []
            for j in range(len(smiles[i])):
                args = {"smi": smiles[i][j]}
                if targets is not None:
                    args['y'] = targets[i, :]
                if j == 0:
                    args['x_d'] = x_d[i, :]
                sublist.append(MoleculeDatapoint.from_smi(**args))
            all_data.append(sublist)
        return np.array(all_data)

    def train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        pl.seed_everything(42)
        # prepare the data
        targets = y_train.values
        all_data = self._get_datapoints(X_train, targets)
        rng = np.random.default_rng(42)
        train_indices = rng.choice(np.arange(all_data.shape[0]), int(all_data.shape[0] * 0.80))
        val_indices = np.array([i for i in range(all_data.shape[0]) if i not in train_indices])

        train_datasets = [MoleculeDataset(all_data[train_indices, i], self.featurizer) for i in range(all_data.shape[1])]
        val_datasets = [MoleculeDataset(all_data[val_indices, i], self.featurizer) for i in range(all_data.shape[1])]
        train_mcdset = MulticomponentDataset(train_datasets)
        train_mcdset.cache = True
        X_d_scaler = train_mcdset.normalize_inputs("X_d")
        val_mcdset = MulticomponentDataset(val_datasets)
        val_mcdset.cache = True
        val_mcdset.normalize_inputs("X_d", X_d_scaler)
        train_loader = build_dataloader(train_mcdset, batch_size=32, num_workers=1, persistent_workers=True)
        val_loader = build_dataloader(val_mcdset, batch_size=32, shuffle=False, num_workers=1, persistent_workers=True)

        # define the model
        chemeleon_mp = torch.load(self.mp_path, weights_only=True)
        mp_blocks = [
            nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
            for _ in range(train_mcdset.n_components)
        ]
        for block in mp_blocks:
            block.load_state_dict(chemeleon_mp["state_dict"])
        mp = nn.MulticomponentMessagePassing(
            mp_blocks, train_mcdset.n_components, shared=True
        )
        agg = nn.MeanAggregation()
        ffn = RegressionFFN(
            n_tasks=targets.shape[1],
            input_dim=mp.output_dim + X_d_scaler[0].mean_.shape[0],
            hidden_dim=2_048,
            n_layers=1,
            # ensure the nn outputs are always identically equal to 1 in sum
            output_transform=torch.nn.Softmax(dim=-1),
        )
        self.model = MulticomponentMPNN(
            message_passing=mp,
            agg=agg,
            predictor=ffn,
            X_d_transform=ScaleTransform.from_standard_scaler(X_d_scaler[0]),
        )

        # fit the model
        outdir = Path("chemprop_output") / datetime.now().strftime("%Y-%d-%m_%H-%M-%S")
        tensorboard_logger = TensorBoardLogger(
            outdir,
            name="tensorboard_logs",
            default_hp_metric=False,
        )
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                verbose=False,
                patience=3,
            ),
            ModelCheckpoint(
                monitor="val_loss",
                dirpath=outdir / "checkpoints",
                save_top_k=1,
                mode="min",
            ),
        ]
        trainer = pl.Trainer(
            max_epochs=20,
            logger=tensorboard_logger,
            log_every_n_steps=1,
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            callbacks=callbacks,
        )

        trainer.fit(self.model, train_loader, val_loader)
        ckpt_path = trainer.checkpoint_callback.best_model_path
        print(f"Reloading best model from checkpoint file: {ckpt_path}")
        self.model = self.model.__class__.load_from_checkpoint(ckpt_path)
        trainer.validate(self.model, val_loader)

    def predict(self, X: pd.DataFrame) -> torch.Tensor:
        all_data = self._get_datapoints(X, None)
        inference_datasets = [MoleculeDataset(all_data[:, i], self.featurizer) for i in range(all_data.shape[1])]
        inference_mcdset = MulticomponentDataset(inference_datasets)
        inference_loader = build_dataloader(inference_mcdset, batch_size=32, shuffle=False)
        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
        )
        return torch.cat(trainer.predict(self.model, dataloaders=inference_loader), dim=0)

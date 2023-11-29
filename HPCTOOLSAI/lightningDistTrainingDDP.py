import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Timer
from pytorch_lightning.callbacks import Callback
from datetime import datetime

class TimeCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        self.start_time = datetime.now()

    def on_train_end(self, trainer, pl_module):
        end_time = datetime.now()
        training_time = end_time - self.start_time
        print(f"Training finished. Total training time: {training_time}")

class MySlurmCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        slurm_id = os.getenv('SLURM_JOB_ID')
        slurm_rank = os.getenv('SLURM_PROCID')
        device_id = torch.cuda.current_device()
        print(f"SLURM_JOB_ID: {slurm_id}, SLURM_PROCID: {slurm_rank}, CUDA Device ID: {device_id}")

class CaliforniaHousing(pl.LightningModule):
    def __init__(self, learning_rate=0.0001):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = torch.nn.Sequential(
            torch.nn.Linear(8, 24),
            torch.nn.ReLU(),
            torch.nn.Linear(24, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, 1)
        )
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def main():
    # Read data and create DataLoader
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize Lightning model
    model = CaliforniaHousing()

    # Set up Trainer for DDP training with callbacks
    trainer = pl.Trainer(
        gpus=2,
        max_epochs=500,
        num_nodes=2,
        accelerator='ddp',
        callbacks=[
            #EarlyStopping(monitor='train_loss'),
            ModelCheckpoint(dirpath='checkpoints/', filename='{epoch}-{train_loss:.2f}'),
            LearningRateMonitor(logging_interval='step'),
            TimeCallback(),
            MySlurmCallback(),
        ]
    )

    # Start training
    print("Starting DDP training")
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    main()
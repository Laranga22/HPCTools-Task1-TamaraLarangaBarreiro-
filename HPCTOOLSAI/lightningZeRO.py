import pytorch_lightning as pl
from pytorch_lightning.plugins import DeepSpeedPlugin
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


# Define el modelo
class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.MSELoss()(y_hat, y)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0001)

def main():
    # Read data and create DataLoader
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # Supongamos que X_train e y_train son tus datos de entrenamiento
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=32)

    # Inicializa el modelo
    model = MyModel()
    n_epochs = 500

    # Inicializa el Trainer con el plugin DeepSpeed para usar ZeRO
    trainer = pl.Trainer(
        gpus=2,
        num_nodes=2,
        precision=16,
        plugins=DeepSpeedPlugin(stage=3),
        max_epochs=n_epochs,
    )

    # Entrena el modelo
    trainer.fit(model, train_loader)
    print(f"GPUs used: {trainer.gpus}")

if __name__ == '__main__':
    main()
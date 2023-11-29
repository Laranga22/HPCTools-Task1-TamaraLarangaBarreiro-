import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision
from torchvision.transforms import Compose, ToTensor, Resize
import pytorch_lightning as pl
import numpy as np

class PatchExtractor(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, input_data):
        batch_size, channels, height, width = input_data.size()
        assert height % self.patch_size == 0 and width % self.patch_size == 0, \
            f"Input height ({height}) and width ({width}) must be divisible by patch size ({self.patch_size})"

        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        num_patches = num_patches_h * num_patches_w

        patches = input_data.unfold(2, self.patch_size, self.patch_size). \
            unfold(3, self.patch_size, self.patch_size). \
            permute(0, 2, 3, 1, 4, 5). \
            contiguous(). \
            view(batch_size, num_patches, -1)

        # Expected shape of a patch on default settings is (4, 196, 768)

        return patches

class InputEmbedding(nn.Module):

    def __init__(self, args):
        super(InputEmbedding, self).__init__()
        self.patch_size = args.patch_size
        self.n_channels = args.n_channels
        self.latent_size = args.latent_size
        self.batch_size = args.batch_size
        self.input_size = self.patch_size * self.patch_size * self.n_channels

        # Linear projection
        self.LinearProjection = nn.Linear(self.input_size, self.latent_size)
        # Class token
        self.class_token = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size))
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size))

    def forward(self, input_data):
        # Patchifying the Image
        patchify = PatchExtractor(patch_size=self.patch_size)
        patches = patchify(input_data)

        linear_projection = self.LinearProjection(patches)
        b, n, _ = linear_projection.shape
        linear_projection = torch.cat((self.class_token, linear_projection), dim=1)
        pos_embed = self.pos_embedding[:, :n + 1, :]
        linear_projection += pos_embed

        return linear_projection

class EncoderBlock(nn.Module):

    def __init__(self, args):
        super(EncoderBlock, self).__init__()

        self.latent_size = args.latent_size
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.norm = nn.LayerNorm(self.latent_size)
        self.attention = nn.MultiheadAttention(self.latent_size, self.num_heads, dropout=self.dropout)
        self.enc_MLP = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size * 4, self.latent_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, emb_patches):
        first_norm = self.norm(emb_patches)
        attention_out = self.attention(first_norm, first_norm, first_norm)[0]
        first_added = attention_out + emb_patches
        second_norm = self.norm(first_added)
        mlp_out = self.enc_MLP(second_norm)
        output = mlp_out + first_added

        return output

class LightningViT(pl.LightningModule):
    class ViT(nn.Module):
        def __init__(self, args):
            super(LightningViT.ViT, self).__init__()
            self.num_encoders = args.num_encoders
            self.latent_size = args.latent_size
            self.num_classes = args.num_classes
            self.dropout = args.dropout

            self.embedding = InputEmbedding(args)
            # Encoder Stack
            self.encoders = nn.ModuleList([EncoderBlock(args) for _ in range(self.num_encoders)])
            self.MLPHead = nn.Sequential(
                nn.LayerNorm(self.latent_size),
                nn.Linear(self.latent_size, self.latent_size),
                nn.Linear(self.latent_size, self.num_classes),
            )

        def forward(self, test_input):
            enc_output = self.embedding(test_input)
            for enc_layer in self.encoders:
                enc_output = enc_layer(enc_output)

            class_token_embed = enc_output[:, 0]
            return self.MLPHead(class_token_embed)

    def __init__(self, args):
        super(LightningViT, self).__init__()
        self.save_hyperparameters(args)
        self.model = self.ViT(args)

    def forward(self, test_input):
        return self.model(test_input)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log('valid_loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer


def main():
    parser = argparse.ArgumentParser(description='Vision Transformer in PyTorch')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--patch-size', type=int, default=16,
                        help='patch size for images (default : 16)')
    parser.add_argument('--latent-size', type=int, default=768,
                        help='latent size (default : 768)')
    parser.add_argument('--n-channels', type=int, default=3,
                        help='number of channels in images (default : 3 for RGB)')
    parser.add_argument('--num-heads', type=int, default=12,
                        help='(default : 16)')
    parser.add_argument('--num-encoders', type=int, default=12,
                        help='number of encoders (default : 12)')
    parser.add_argument('--dropout', type=int, default=0.1,
                        help='dropout value (default : 0.1)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='image size to be reshaped to (default : 224')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='number of classes in dataset (default : 10 for CIFAR10)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs (default : 2)')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='base learning rate (default : 0.01)')
    parser.add_argument('--weight-decay', type=int, default=3e-2,
                        help='weight decay value (default : 0.03)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='batch size (default : 4)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    args = parser.parse_args()

    transforms = Compose([
        Resize((args.img_size, args.img_size)),
        ToTensor()
    ])
    train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transforms)
    valid_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transforms)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)

    model = LightningViT(args)

    trainer = pl.Trainer(
        gpus=2,  # Número de GPUs que quieres usar
        num_nodes=2,  # Número de nodos (máquinas) que estás utilizando
        max_epochs=args.epochs,
        accelerator='ddp',
    )

    trainer.fit(model, train_loader, valid_loader)

if __name__ == "__main__":
    main()   
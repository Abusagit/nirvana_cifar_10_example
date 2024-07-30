import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import Compose, ToTensor, Resize
from torch import optim
import numpy as np
from torch.hub import tqdm

from json import load, dump

from dataclasses import dataclass

#################
# Nirvana funcs #
#################

from nirvana_utils import copy_snapshot_to_out, copy_out_to_snapshot

CONFIG_FILE = "CONFIG.json"
JSON_RESULTS_FILENAME = "result_metrics.json"

@dataclass
class Config:
    patch_size: int = 16
    latent_size: int = 384
    num_heads: int = 6
    epochs: int = 10
    
    
def get_config() -> Config:
    with open(CONFIG_FILE) as f_read:
        config_json = load(f_read)
    config = Config(**config_json)
    
    return config

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

    def __init__(self, args, patch_size, latent_size):
        super(InputEmbedding, self).__init__()
        self.patch_size = patch_size
        self.n_channels = args.n_channels
        self.latent_size = latent_size
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.batch_size = args.batch_size
        self.input_size = self.patch_size * self.patch_size * self.n_channels

        # Linear projection
        self.LinearProjection = nn.Linear(self.input_size, self.latent_size)
        # Class token
        self.class_token = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)

    def forward(self, input_data):
        input_data = input_data.to(self.device)
        # Patchifying the Image
        patchify = PatchExtractor(patch_size=self.patch_size)
        patches = patchify(input_data)

        linear_projection = self.LinearProjection(patches).to(self.device)
        b, n, _ = linear_projection.shape
        linear_projection = torch.cat((self.class_token, linear_projection), dim=1)
        pos_embed = self.pos_embedding[:, :n + 1, :]
        linear_projection += pos_embed

        return linear_projection


class EncoderBlock(nn.Module):

    def __init__(self, args, latent_size, num_heads):
        super(EncoderBlock, self).__init__()

        self.latent_size = latent_size
        self.num_heads = num_heads
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


class ViT(nn.Module):
    def __init__(self, args, latent_size, patch_size, num_heads):
        super(ViT, self).__init__()

        self.num_encoders = args.num_encoders
        self.latent_size = latent_size
        self.num_classes = args.num_classes
        self.dropout = args.dropout

        self.embedding = InputEmbedding(args, patch_size, latent_size)
        # Encoder Stack
        self.encoders = nn.ModuleList([EncoderBlock(args, latent_size, num_heads) for _ in range(self.num_encoders)])
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


class TrainEval:

    def __init__(self, args, epochs, model, train_dataloader, val_dataloader, optimizer, criterion, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch = epochs
        self.device = device
        self.args = args

    def train_fn(self, current_epoch):
        self.model.train()
        total_loss = 0.0
        tk = tqdm(self.train_dataloader, desc="EPOCH" + " [TRAIN] " + str(current_epoch + 1) + "/" + str(self.epoch))

        for t, data in enumerate(tk):
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            tk.set_postfix({"Loss": "%6f" % float(total_loss / (t + 1))})
            if self.args.dry_run:
                break

        return total_loss / len(self.train_dataloader)

    def eval_fn(self, current_epoch, is_test=False):
        self.model.eval()
        total_loss = 0.0
        test_val_token = "[TEST]" if is_test else "[VALID]"
        
        description_string = f"EPOCH {test_val_token} {current_epoch}/{self.epoch}"
        
        tk = tqdm(self.val_dataloader, desc=description_string)

        for t, data in enumerate(tk):
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            tk.set_postfix({"Loss": "%6f" % float(total_loss / (t + 1))})
            if self.args.dry_run:
                break

        return total_loss / len(self.val_dataloader)

    def train(self):
        best_valid_loss = np.inf
        best_train_loss = np.inf
        for i in range(self.epoch):
            train_loss = self.train_fn(i)
            val_loss = self.eval_fn(i)

            if val_loss < best_valid_loss:
                torch.save(self.model.state_dict(), "checkpoints/best-weights.pt")
                print("Saved Best Weights")
                best_valid_loss = val_loss
                best_train_loss = train_loss

                #############
                # IMPORTANT #
                #############

                copy_out_to_snapshot("checkpoints")

        print(f"Training Loss : {best_train_loss}")
        print(f"Valid Loss : {best_valid_loss}")
        
        return best_train_loss, best_valid_loss

    '''
        On default settings:
        
        Training Loss : 2.3081023390197752
        Valid Loss : 2.302861615943909
        
        However, this score is not competitive compared to the 
        high results in the original paper, which were achieved 
        through pre-training on JFT-300M dataset, then fine-tuning 
        it on the target dataset. To improve the model quality 
        without pre-training, we could try training for more epochs, 
        using more Transformer layers, resizing images or changing 
        patch size,
    '''


def main():
    parser = argparse.ArgumentParser(description='Vision Transformer in PyTorch')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # parser.add_argument('--patch-size', type=int, default=16,
    #                     help='patch size for images (default : 16)')
    # parser.add_argument('--latent-size', type=int, default=384,
    #                     help='latent size (default : 384)')
    parser.add_argument('--n-channels', type=int, default=3,
                        help='number of channels in images (default : 3 for RGB)')
    # parser.add_argument('--num-heads', type=int, default=6,
    #                     help='(default : 16)')
    parser.add_argument('--num-encoders', type=int, default=12,
                        help='number of encoders (default : 12)')
    parser.add_argument('--dropout', type=int, default=0.1,
                        help='dropout value (default : 0.1)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='image size to be reshaped to (default : 224')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='number of classes in dataset (default : 10 for CIFAR10)')
    # parser.add_argument('--epochs', type=int, default=2,
    #                     help='number of epochs (default : 2)')
    parser.add_argument('--lr', type=int, default=1e-2,
                        help='base learning rate (default : 0.01)')
    parser.add_argument('--weight-decay', type=int, default=3e-2,
                        help='weight decay value (default : 0.03)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size (default : 16)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train",
                        help="Behaviour of the workflow")
    args = parser.parse_args()
    
    
    config = get_config()
   
    #############
    # IMPORTANT #
    #############

    copy_snapshot_to_out("checkpoints")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Device is {device}")

    transforms = Compose([
        Resize((args.img_size, args.img_size)),
        ToTensor()
    ])
    
    if args.mode == "train":
        
        train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=False, transform=transforms)
        valid_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transforms)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=4, shuffle=False)

        model = ViT(args, latent_size=config.latent_size, num_heads=config.num_heads, patch_size=config.patch_size).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()

        train_loss, valid_loss = TrainEval(args, config.epochs, model, train_loader, valid_loader, optimizer, criterion, device).train()
        test_loss = None

    else:
        
        test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transforms)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=4, shuffle=False)

        model = ViT(args, latent_size=config.latent_size, num_heads=config.num_heads, patch_size=config.patch_size).to(device)
        
        model.load_state_dict(torch.load("checkpoints/best-weights.pt", map_location=device))
        print("Loaded best weights from the training from the location checkpoints/best-weights.pt")
        
        # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        train_loss, valid_loss = None, None
        test_loss = TrainEval(args, config.epochs, model, None, test_loader, None, criterion, device).eval_fn(1)
        
    result_dict = dict(
        train_loss=train_loss,
        validation_loss=valid_loss,
        test_loss=test_loss,
    )
    
    ## dumping result:
    with open(JSON_RESULTS_FILENAME, "w") as f_write:
        dump(obj=result_dict, fp=f_write, indent=4, sort_keys=True)
    
    print(f"Resulting metrics were saved to {JSON_RESULTS_FILENAME}")
    
    copy_out_to_snapshot("checkpoints", dump=True)
    
if __name__ == "__main__":
    main()

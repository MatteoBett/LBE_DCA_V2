import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from numpy import mean


VOCAB = {'A': 0, 'C': 1, 'G': 2, 'U': 3, '-': 4}

class RNA_Attention(nn.Module):
    def __init__(self, vocab_size=5, seq_len=512, hidden_dim=32, device=None):
        super().__init__()
        self.embed = nn.Linear(vocab_size, hidden_dim)  # per-token projection
        self.pos_embedding = nn.Parameter(torch.randn(seq_len, hidden_dim))
        self.attn = nn.Linear(hidden_dim, 1)             # attention scores
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.to(device)

    def forward(self, x, mask):
        # x: (B, L, vocab_size), mask: (B, L)
        # h = self.embed(x)              # (B, L, D)
        h = self.embed(x) + self.pos_embedding[:x.size(1)]  # (B, L, D)
        attn_scores = self.attn(h).squeeze(-1)  # (B, L)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, L)
        context = (attn_weights.unsqueeze(-1) * h).sum(dim=1)  # (B, D)
        return self.output(context).squeeze(-1)  # (B,)


class RNA_Conv1D(nn.Module):
    def __init__(self, vocab_size=5, hidden_dim=16, num_layers=3,
                 kernel_size=3, seq_len=512, device=None):
        super().__init__()
        in_ch = vocab_size

        layers = []
        for i in range(num_layers):
            d = 2 ** i
            layers += [nn.Conv1d(in_ch, hidden_dim, kernel_size, padding=d, dilation=d),
                       nn.ReLU()]
            in_ch = hidden_dim
        self.conv_net = nn.Sequential(*layers)

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim*seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.to(device)

    def forward(self, x, mask):
        x = x.transpose(1, 2)  # (B, C, L)
        x = self.conv_net(x)
        x = x * (~mask.unsqueeze(1)).to(x.dtype)  # mask padding
        # x = x.sum(dim=2) / (~mask).sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)  # (B, C)
        return self.output_head(x.flatten(1)).squeeze(-1)  # output_head: Linear(C, 1)


class RNA_MLP(nn.Module):
    def __init__(self, vocab_size=5, seq_len=512, hidden_dim=128, num_layers=3, device=None):
        super().__init__()
        input_dim = vocab_size * seq_len

        layers = []
        for _ in range(num_layers):
            layers += [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)
        self.to(device)

    def forward(self, x, mask):
        # x: (B, L, vocab_size), mask: (B, L) → mask is unused
        x = x.flatten(1)  # (B, L * vocab_size)
        return self.mlp(x).squeeze(-1)  # (B,)


class RNA_reg(nn.Module):
    def __init__(self, vocab_size=5, seq_len=512, device=None):
        super().__init__()
        self.sig = nn.Sigmoid()
        self.parms = nn.Linear(vocab_size * seq_len, 1)
        self.to(device)

    def forward(self, x, mask):
        # x: (B, L, vocab_size), mask: (B, L) → mask is unused
        x = x.flatten(1)  # (B, L * vocab_size)
        return self.sig(self.parms(x.flatten(1))).squeeze(-1)  # (B,)


class RNADataset(Dataset):
    def __init__(self, sequences, targets, pad_token='-'):
        """
        sequences: list of str (RNA sequences)
        targets: list of float
        vocab: dict mapping chars to int (e.g. {'A':0, 'C':1, 'G':2, 'U':3, '-':4})
        pad_token: character used for padding
        """
        self.pad_id = VOCAB[pad_token]
        self.sequences = sequences
        self.targets = torch.tensor(targets, dtype=torch.float)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = torch.tensor([VOCAB[c] for c in seq], dtype=torch.long)  # (L,)
        one_hot = F.one_hot(x, num_classes=len(VOCAB)).float()       # (L, q)
        mask = (x == self.pad_id)                                         # (L,)
        return one_hot, mask, self.targets[idx]

def train(model, train_loader, val_loader, optimizer, loss_fn, max_epochs=100, patience=5, device='cpu'):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = model.state_dict()  # ensure it's always defined
    model.to(device)

    for epoch in range(max_epochs):
        model.train()
        tmp = []
        for x, mask, y in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            loss = loss_fn(model(x, mask), y)
            tmp += [loss.item()]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print(epoch, mean(tmp))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, mask, y in val_loader:
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                val_loss += loss_fn(model(x, mask), y).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                model.load_state_dict(best_model)
                break

def prepare_train_val(sequences, targets, pad_token='-',
                      batch_size=32, val_frac=0.1, seed=42):
    dataset = RNADataset(sequences, targets, pad_token)
    n = len(dataset)
    n_val = int(n * val_frac)
    n_train = n - n_val

    torch.manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    return train_loader, val_loader
                                          
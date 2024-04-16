# --------------------------------------
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# --------------------------------------
torch.manual_seed(1337)
block_size = 8
batch_size = 4
max_iters = 4000
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32
epsilon = 1e-5

# --------------------------------------
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# open and file
filename = "input.txt"
with open(filename, "r", encoding="utf-8") as f:
    text = f.read()

# get the vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create encoder and decoder
itos = {i: ch for i, ch in enumerate(chars)}
stoi = {ch: i for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]  # take a string and output a list of ints
decode = lambda lst: "".join(
    [itos[i] for i in lst]
)  # take a list of ints and output a string


# encode the whole data to a torch tensor
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if (split == "train") else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    xb = torch.stack([data[i : i + block_size] for i in ix])
    yb = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb


@torch.no_grad()
def evaluate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    # one head of self-attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def __call__(self, x):
        B, T, C = x.shape  # (B, T, C)
        k = self.key(x)  # (B, T, 16) head-size = 16
        q = self.query(x)  # (B, T, 16) head-size = 16

        # dimensions which are swapped here -1, and -2
        # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
        # normalize gaussian
        wei = q @ k.transpose(-2, -1) * C ** (-0.5)
        # using softmax
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """a simple linear layer
    followed by non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa_head = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)  # (?? n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, num_heads=4),
            Block(n_embd, num_heads=4),
            Block(n_embd, num_heads=4),
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both of size (B, T)
        # for traing when `targets` is None
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (B, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        # print(f"{x.shape = }")
        # print(f"{n_embd = }")
        # logits = self.lm_head(tok_emb) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # for testing when `targets` is not None
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def func_test(self, idx):
        out_self = self(idx)
        out_self_forward = self.forward(idx)
        equal = (
            torch.equal(out_self[0], out_self_forward[0])
            and (out_self[1] is None)
            and (out_self_forward[1] is None)
        )
        return out_self, out_self_forward, equal

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context ?!
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)  # the same as self(idx)
            _logits = logits[:, -1, :]  # pick the last
            _probs = F.softmax(_logits, dim=1)
            idx_next = torch.multinomial(input=_probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# define model
model = BigramLanguageModel()
model = model.to(device)
# pytorch optimization
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = evaluate_loss()
        print(
            f'step {iter:4d} train loss {losses["train"]:.04f}, '
            f'val loss {losses["val"]:.4f}'
        )

    # sample data
    xb, yb = get_batch("train")

    # evaluate loss
    logits, loss = model(xb, yb)

    # backward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = decode(model.generate(context, max_new_tokens=500).tolist()[0])
print(out)

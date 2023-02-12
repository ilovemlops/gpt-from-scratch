"""simplest baeline: bigram language model - version 2"""
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel to train transformer
block_size = 8 # maximum context length for predictions
max_iters = 50000
eval_interval = 5000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32


torch.manual_seed(137)

# load input data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# get all characters in dataset and sort them
chars = sorted(list(set(text)))
# get vocabularly size
vocab_size = len(chars)

# build encoder and decoder on character level
stoi = {ch:i for i,ch in enumerate(chars)} # string to integer
itos = {i:ch for i, ch in enumerate(chars)} # integer to string

encode = lambda s: [stoi[c] for c in s] # encoder: takes a string and outputs list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: takes a list of integers and outputs a string

# encode entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
# split data into train and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

@torch.no_grad() # context manager - what happens inside would not be called .backward
def estimate_loss():
    out = {}
    model.eval() # evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # training mode
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('trill', torch.tril(torch.ones(block_size,block_size)) )

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C) , head_size
        q = self.query(x) # (B, T, C)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) ---> (B, T, T)
        wei = wei.masked_fill(self.trill[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, C) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

# bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # C - number of channels
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(4, n_embd//4)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C) , n_embd
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T,C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.sa_heads(x) # apply one head of self-attention (B, T, C)
        logits = self.lm_head(x) # (B, T, C) , vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # one dimensional array to stretch an arrayy
            targets = targets.view(B*T) # or can be set to -1
            # negative logloss
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond) # goes to forward
            # take the last time step
            logits = logits[:,-1,:] # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):

    # evaluate loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    # zero gradients from previous step
    optimizer.zero_grad(set_to_none=True)
    # get gradients for all parameters
    loss.backward()
    # use gradients to update parameters
    optimizer.step()

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

torch.save(model, "model.pkl")
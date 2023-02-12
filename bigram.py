"""simplest baeline: bigram language model"""
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel to train transformer
block_size = 8 # maximum context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200


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

# bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # C - number of channels

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)
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
            # get the predictions
            logits, loss = self(idx) # goes to forward
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
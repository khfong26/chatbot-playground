import torch
import torch.nn.functional as F
from torch import nn
import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("grafstor/simple-dialogs-for-chatbot")
files = os.listdir(path)
with open(os.path.join(path, files[0]), 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)
n_embd = 384  # Size of the embedding vectors
max_iterations = 1
eval_interval = 10
learning_rate = 3e-4
batch_size = 64  # Number of sequences in a batch
block_size = 256  # Size of the context window
eval_iters = 200
n_head = 6  # Number of attention heads
n_layer = 6  # Number of transformer blocks
dropout = 0.2  # Dropout rate


# Tokenizer
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[s] for s in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode the text data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # 90% for training
train_data = data[:n]
val_data = data[n:]

# Function to get a batch of data
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = m(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) 
        q = self.query(x)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList((Head(head_size) for _ in range(n_heads)))
        self.proj = nn.Linear(n_embd, n_embd)  # Projection layer to combine heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate outputs from all heads
        out = self.proj(out)  # Project the concatenated output
        return out  # Return the final output after projection
    
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd),  # Output layer to match the input dimension
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])  # Stack n_layer blocks
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)  # Apply final layer normalization
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T) 
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        else:
            return logits, None
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # Get the last block_size tokens
            logits, loss = self(idx_cond) # predict next token
            logits = logits[:, -1, :] # focus on the last time step
            probs = F.softmax(logits, dim = -1) # convert to probabilities
            idx_next = torch.multinomial(probs, num_samples = 1) # sample from the distribution
            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to the input
        return idx

m = BigramLanguageModel()
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

print("hello")

eval_interval = 1

for step in range(5):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)  # Forward pass
    optimizer.zero_grad(set_to_none=True)  # Zero gradients
    loss.backward()  # Backward pass
    optimizer.step()  # Update parameters

    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {step}, Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")

# generate text starting from an empty space
idx = torch.zeros((1,1), dtype=torch.long)  # Start with a single token
print(decode(m.generate(idx, max_new_tokens=1000)[0].tolist()))  # Generate text and decode it


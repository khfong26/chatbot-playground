import torch
from torch import nn
from torch.nn import functional as F
import kagglehub
import os


block_size = 8  # Size of the context window
batch_size = 32  # Number of sequences in a batch
eval_batch_size = 32  # Define evaluation batch size
learning_rate = 1e-3  # Learning rate for the optimizer
iterations = 30000  # Number of training iterations
iter_checks = 1000  # Frequency of loss checks during training


# Download latest version
path = kagglehub.dataset_download("grafstor/simple-dialogs-for-chatbot")
files = os.listdir(path)
with open(os.path.join(path, files[0]), 'r', encoding='utf-8') as f:
    text = f.read()

# Extract unique characters and create vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create character to index and index to character mappings
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


# Function to estimate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_batch_size)
        for k in range(eval_batch_size):
            xb, yb = get_batch(split)
            logits, loss = m(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


# Define the Bigram Language Model
class BigramLanguageModel(nn.Module):
    # Initialize the model with a vocabulary size
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # Forward pass through the model
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B, T, C)
        
        # If targets are provided, compute the loss
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T) 
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        else:
            return logits, None
        
    # Generate new tokens based on the input index
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx) # predict next token
            logits = logits[:, -1, :] # focus on the last time step
            probs = F.softmax(logits, dim = -1) # convert to probabilities
            idx_next = torch.multinomial(probs, num_samples = 1) # sample from the distribution
            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to the input
        return idx

# Initialize the model, and optimizer
m = BigramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(m.parameters(), lr= learning_rate)


for step in range(iterations):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)  # Forward pass
    optimizer.zero_grad(set_to_none=True)  # Zero gradients
    loss.backward()  # Backward pass
    optimizer.step()  # Update parameters

    if step % iter_checks == 0:
        losses = estimate_loss()
        print(f"Step {step}, Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")


# Save the model state
torch.save(m.state_dict(), "bigram_model.pth")


# Generate text using the trained model
idx = torch.zeros((1,1), dtype=torch.long)  # Start with a single token
print(decode(m.generate(idx, max_new_tokens=1000)[0].tolist()))  # Generate text and decode it



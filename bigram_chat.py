import torch
from torch import nn
from torch.nn import functional as F
import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("grafstor/simple-dialogs-for-chatbot")
files = os.listdir(path)
with open(os.path.join(path, files[0]), 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
decode = lambda l: ''.join([itos[i] for i in l])

# Define the model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
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
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Load the trained model
m = BigramLanguageModel(vocab_size)
m.load_state_dict(torch.load("bigram_model.pth"))
m.eval()

# Start chatting
while True:
    user_input = input("You: ")
    print("----------")
    if not user_input:
        break
    idx = torch.tensor([[stoi.get(ch, 0) for ch in user_input]], dtype=torch.long)
    generated = m.generate(idx, max_new_tokens=200)
    print("Bot:", decode(generated[0].tolist()))
    print("----------")

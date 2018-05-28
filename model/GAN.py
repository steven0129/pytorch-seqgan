import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inp, hidden):
        embedding = self.embeddings(inp)                        # batch_size * embedding_dim
        embedding = embedding.view(1, -1, self.embedding_dim)   # 1 * batch_size * embedding_dim
        out, hidden = self.gru(embedding, hidden)               # 1 * batch_size * hiddem_dim (out)
        out = self.out(out.view(-1, self.hidden_dim))           # batch_size * vocab_size
        out = F.log_softmax(out)                                
        return out, hidden
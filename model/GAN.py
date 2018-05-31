import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.gpu = gpu

    def forward(self, inp, hidden):
        embedding = self.embeddings(inp)                        # batch_size * embedding_dim
        embedding = embedding.view(1, -1, self.embedding_dim)   # 1 * batch_size * embedding_dim
        out, hidden = self.gru(embedding, hidden)               # 1 * batch_size * hiddem_dim (out)
        out = self.out(out.view(-1, self.hidden_dim))           # batch_size * vocab_size
        out = F.log_softmax(out)
        return out, hidden

    def NLLLoss(self, inp, target):
        """
        Returns the NLL Loss for predicting target sequence

        Inputs: inp, target
            - inp: batch_size * seq_len
            - target: batch_size * seq_len

            inp should be target with <s> (start letter) prepended
        """

        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)         # seq_len * batch_size
        target = target.permute(1, 0)   # seq_len * batch_size

        # init hidden layer
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))
        if self.gpu: h = h.cuda()

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            loss += nn.NLLLoss()(out, target[i])

        return loss
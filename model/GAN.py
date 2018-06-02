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
    
    def sample(self, num, start_symbol):
        samples = torch.zeros(num, self.max_seq_len).long()
        h = autograd.Variable(torch.zeros(1, num, self.hidden_dim))
        inp = autograd.Variable(torch.LongTensor([start_symbol]*num))

        if self.gpu:
            h = h.cuda()
            samples = samples.cuda()
            inp = inp.cuda()

        for i in range(self.max_seq_len):
            out, h = self.forward(inp, h)               # out: num * vocab_size
            out = torch.multinomial(torch.exp(out), 1)  # num * 1
            for j in range(num):
                samples[j, i] = out.data[j]
            
            inp = out.view(-1)

        return samples

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

class Discriminator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, dropout=0.2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2 * 2 * hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def forward(self, input, hidden):
        # input dim                                                # batch_size * seq_len
        emb = self.embeddings(input)                               # batch_size * seq_len * embedding_dim
        emb = emb.permute(1, 0, 2)                                 # seq_len * batch_size * embedding_dim
        _, hidden = self.gru(emb, hidden)                          # 4 * batch_size * hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size * 4 * hidden_dim
        out = self.gru2hidden(hidden.view(-1, 4 * self.hidden_dim))  # batch_size * 4 * hidden_dim
        out = F.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)                                 # batch_size * 1
        out = F.sigmoid(out)
        return out

    def train(self, pos_samples, neg_samples, d_steps, epochs):
        pass
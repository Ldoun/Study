import torch
import torch.nn as nn 
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import simple_mnt.data_loader as data_loader
from simple_mnt.search import SingleBeamSearchBoard

class Attention(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()

        self.linear = nn.Linear(hidden_size,hidden_size,bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,h_src,h_t_tgt,mask = None):
        #|h_src| = (batch_size,length,hidden_size)
        #|h_t_tgt| = (batch_size,1,hidden_size)
        #|mask| = (batch_size,length)

        query = self.linear(h_t_tgt.squeeze(1)).unsqueeze(-1)
        #|query| = (batch_size,hidden_size,1)

        weight = torch.bmm(h_src,query).squeeze(-1)
        #|weight| = (batch_size,length)

        if mask is not None:
            weight.maked_fill_(mask,-float('inf'))
        weight = self.softmax(weight)
        #현재 timestep에 영향을 많이 끼치는 timestep(단어)의 소프트맥스 확률이 크게 나타남

        context_vector = torch.bmm(weight.unsqueeze(1),h_src) 
        #|context_vector| = (batch_size,1,hidden_size)

        return context_vector

class Encoder(nn.Module):
    def __init__(self,word_vec_dim,hidden_size,n_layers=4,dropout_p=.2):
        super().__init__()
        self.rnn = nn.LSTM(
            word_vec_dim,
            int(hidden_size/2), #bidirectional 이므로 hidden size = hs/2
            num_layers = n_layers,
            dropout = dropout_p,
            bidirectional = True,
            batch_first = True
        )

    def forward(self,emb):
        #|emb| = (batch_size,length,word_vec_dim)
        if isinstance(emb,tuple):
            x,lengths = emb
            x = pack(x,lengths.tolist(),batch_first=True)
            #최대 length size에 맞게 input들 pad하는 과정   

        else:
            x = emb

        y,h = self.rnn(x)
        #|y| = (batch_size,length,hidden_size)
        #|h[0]| = (num_layers * 2,batch_size, hidden_size / 2)

        if isinstance(emb,tuple):
            y,_ = unpack(y,batch_first=True)
            #unpack 각 batch 별 개별 length가짐

        return y,h

class Decoder(nn.Module):
    def __init__(self,word_vec_size,hidden_size,n_layers=4,dropout_p=.2):
        super().__init__()

        self.rnn = nn.LSTM(
            word_vec_size + hidden_size,
            hidden_size,
            num_layers = n_layers,
            dropout = dropout_p,
            bidirectional = False,
            batch_first = True,
        )

    def forward(self,emb_t,h_t_1_tilde,h_t_1):
        #|emb_t| = (batch_size,1,word_vec_size)
        #|h_t_1_tilde| = (batch_size,1,hidden_size)
        #|h_t_1_[0]| = (n_layers,batch_size,hidden_size)

        batch_size = emb_t.size(0)
        hidden_size = h_t_1[0].size(-1)

        if h_t_1_tilde is None:
            #first time-step
            h_t_1_tilde = emb_t.new(batch_size,1,hidden_size).zero_()
            #해당 tensor와 같은 type,디바이스로 tensor를 만들어주는 함수

        x = torch.cat([emb_t,h_t_1_tilde],dim = -1)
        #|x| = (batch_size,1,word_vec_size + hidden_size)

        y,h = self.rnn(x,h_t_1)
        #|y| = (batch_size,1,hidden_size)
        #|h[0]| = (num_layers,batch_size,hidden_size)

        return y,h

class Generator(nn.Module):
    def __init__(self,hidden_size,output_size):
        super().__init__()

        self.output = nn.Linear(hidden_size,output_size = output_size) 
        self.softmax = nn.Softmax(dim = -1) #cross-entrophy 사용

    def forward(self,x):
        y = self.softmax(self.output(x))
        #|y| = (batch_size,length,ouput_size(vocab_size))
        return y

class Seq2Seq(nn.Module):
    def __init__(
        self,
        input_size,
        word_vec_size,
        hidden_size,
        output_size,
        n_layers = 4,
        dropout_p = .2
    ):
        self.input_size = input_size #|vodcab|
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.output_size = output_size #|vodcab|
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.emb_src = nn.Embedding(input_size,word_vec_size)
        self.emb_dec = nn.Embedding(output_size,word_vec_size)

        self.encoder = Encoder(
            word_vec_dim=word_vec_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout_p=dropout_p
        )

        self.decoder = Decoder(
            word_vec_size=word_vec_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout_p=dropout_p
        )

        self.attn = Attention(hidden_size=hidden_size)

        self.concat = nn.Linear(hidden_size*2,hidden_size)
        self.tanh = nn.Tanh()
        self.generator = Generator(hidden_size=hidden_size,output_size=output_size)

    def generate_mask(self,x,length):
        mask = []

        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                mask += [torch.cat([x.new_ones(1,l).zero_(),
                                    x.new_ones(1,(max_length - l))]
                                    ,dim = -1)]

            else:
                mask += [x.new_ones(1,l).zero_()]

        mask = torch.cat(mask,dim=0).bool()
        return mask
    
    def merge_encoder_hiddens(self,encoder_hidden):
        new_hidden = []
        new_cells = []

        hidden,cells = encoder_hidden

        for i in range(0,hidden.size(0),2):
            new_hidden += [torch.cat([hidden[i],hidden[i+1]],dim=-1)]
            new_cells += [torch.cat([cells[i],cells[i+1]],dim=-1)]

        new_hidden,new_cells = torch.stack(new_hidden),torch.stack(new_cells)

        return (new_hidden,new_cells)

    def fast_merge_encoder_hidden(self,encoder_hiddens):
        h_0_tgt, c_0_tht = encoder_hiddens
        batch_size = h_0_tgt.size(1)

        h_0_tgt = h_0_tgt.transpose(0,1).contiguous().view(batch_size,-1,self.hidden_size).transpose(0,1).contiguous()
        c_0_tgt = c_0_tgt.transpose(0,1).contiguous().view(batch_size,-1,self.hidden_size).transpose(0,1).contiguous()

        return h_0_tgt,c_0_tgt

    def forward(self,src,tgt):
        #|src| = (batch_size,n,vocab) 번역 전 문장
        #|tgt| = (batch_size,n,vocab) 번역 후 문장
        batch_size = tgt.size(0)

        mask = None
        x_length = None
        if isinstance(src,tuple):
            x,x_length = src
            mask = self.generate_mask(x,x_length)
            #|mask| = (batch_size,length)

        else:
            x = src

        if isinstance(tgt,tuple):
            tgt = tgt[0]

        emb_src = self.emb_src(x)
        #|emb_src| = (batch_size,length,word_vec_size)

        h_src, h_0_tgt = self.encoder((emb_src,x_length))
        #|h_src| = (batch_size,length,hidden_size)
        #|h_0_tgt| = (num_layers * 2,batch_size, hidden_size / 2)

        h_0_tgt = self.fast_merge_encoder_hidden(h_0_tgt)
        emb_tgt = self.emb_dec(tgt)
        #|emb_tgt| = (batch_size,length,word_vec_size)

        h_tilde = []

        h_t_tilde = None
        decoder_hidden = h_0_tgt
        
        for t in range(tgt.size(1)):
            emb_t = emb_tgt[:,t,:].unsqueeze(1)
            #|emb_t| = (batch_size,1,word_vec_size)
            #|h_t_tilde| = (batch_size,1,hidden_size)

            decoder_output,decoder_hidden = self.decoder(emb_t,h_t_tilde,decoder_hidden)

            #|decoder_output| = (batch_size,1,hidden_size)
            #|decoder_hidden| = (n_layer,batch_size,hidden_size)

            context_vector = self.attn(h_src,decoder_output,mask)

            h_t_tilde = self.tanh(self.concat(torch.cat([decoder_output,context_vector],dim=-1)))
            #|h_t_tilde| = (batch_size,1,hidden_size)

            h_tilde += [h_t_tilde]

        h_tilde = torch.cat(h_tilde,dim=1)
        #|h_t_tilde| = (batch_size,length,hidden_size)

        y_hat = self.generator(h_t_tilde)
        #|y_hat| = (batch_size,length,ouput_size)

        return y_hat



    def search(self,):
        pass
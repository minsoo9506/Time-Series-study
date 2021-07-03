# https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/models/deepar/__init__.py
# https://github.com/zhykoties/TimeSeries/blob/master/model/net.py
# 김기현 github

import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, input_size: int = 6, # 근데 여기 multivariate할꺼면 바꿔야할듯
                       hidden_size: int = 32,
                       num_layers: int = 3,
                       dropout_p: float = 0.2):
        super().__init__()

        
        self.rnn = nn.LSTM(
            input_size = input_size, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            dropout  = dropout_p,
            batch_first = True
        )
        
    def forward(self, emb):
        # |emb| = (batch_size, seq_len, input_dim)
        _, h = self.rnn(emb)
        # |h[0]| = (num_layers, batch_size, hidden_size)
        return h

class Decoder(nn.Module):
    
    def __init__(self, input_size: int = 5,
                       hidden_size: int = 32,
                       num_layers: int = 3,
                       dropout_p: float = 0.2):
        super().__init__()
        
        self.rnn = nn.LSTM(
            input_size = input_size + 1, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            dropout  = dropout_p,
            batch_first = True
        )
        
    def forward(self, emb_t, h_t_1_tilde, h_t_1):
        # h_t_1_tilde : 이전 time-step에서의 output
        # h_t_1 = (h,c) : 이전 time-step의 hidden, cell state
        # |emb_t| = (batch_size, 1, input_dim-1) 
        # -> encoder에서는 input에 y값과 covariate값을 한번에 넣었고
        #    decoder에서는 구조상 (teacher forcing과 inference) 구분한다
        #    그래서 input_dim - 1 이라고 표시
        # |h_t_1_tilde| = (batch_size, 1, 1)
        # |h_t_1[0]| = (num_layers, batch_size, hidden_size)
        batch_size = emb_t.size(0)
        hidden_size = h_t_1[0].size(-1)

        # 만약에 decoder에서 first time-step이라면
        if h_t_1_tilde is None:
            h_t_1_tilde = emb_t.new(batch_size, 1, 1).zero_()
        
        x = torch.cat([emb_t, h_t_1_tilde], dim=-1)

        y, h = self.rnn(x, h_t_1)
        # |y| = (batch_size, 1, hidden_size)
        # |h[0]| = (num_layers, batch_size, hidden_size)

        return y, h

class GenerateFromNormal(nn.Module):

    def __init__(self, hidden_size: int = 32):
        super().__init__()

        self.distribution_mu = nn.Linear(hidden_size, 1)
        self.distribution_sigma = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # |x| = (batch_size, 1, hidden_size)
        x = torch.squeeze(x, dim=1)
        # |x| = (batch_size, hidden_size)

        mu = self.distribution_mu(x)
        # |mu| = (batch_size, 1)
        sigma = nn.Softplus(self.distribution_sigma(x))
        # |sigma| = (batch_size, 1)
        batch_size = sigma.shape[0]
        error = torch.normal(mean=0, std=1, size=(batch_size, 1))

        y_hat = mu + sigma * error

        y_hat = torch.squeeze(y_hat, dim=1)
        sigma = torch.squeeze(sigma, dim=1)
        # |y_hat| = (batch_size, )
        # |sigma| = (batch_size, )

        return y_hat, sigma

class DeepAR(nn.Module):
    def __init__(self, 
                 batch_size: int = 64,
                 pred_seq_len: int = 24,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.pred_seq_len = pred_seq_len

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.generateFromNormal = GenerateFromNormal()

    def forward(self, enc_x, dec_emb, dec_teacher):
        # |enc_x| = (batch_size, seq_len, input_dim)
        # 주의!
            # dec에 들어가는 input들은 t시점을 예측하기위해
            # t-1시점의 data가 들어가야 한다 (emb, teacher 모두)
        # |dec_emb| = (batch_size, pred_seq_len, input_dim-1)
        # |dec_teacher| = (batch_size, pred_seq_len, 1)

        outputs = enc_x.new_empty((self.batch_size, self.pred_seq_len))
        enc_h = self.encoder(enc_x)

        # teacher forcing
        emb_t, h_t_1_tilde, h_t_1 = dec_emb[:,0,:], dec_teacher[:,0,:], enc_h
        for t in range(1, self.pred_seq_len+1):
            dec_y, dec_h = self.decoder(emb_t, h_t_1_tilde, h_t_1)
            # final outputs
            outputs[:, t-1], _ = self.generateFromNormal(dec_y)
            
            if t == self.pred_seq_len:
                break
            # decoder input update
            emb_t = dec_emb[:,t,:]
            h_t_1_tilde = dec_teacher[:,t,:]
            h_t_1 = dec_h
            
        return outputs

    def inference(self, enc_x, dec_emb):
        # |enc_x| = (batch_size, seq_len, input_dim)
        # |dec_emb| = (batch_size, pred_seq_len, input_dim-1)
        
        outputs = enc_x.new_empty((self.batch_size, self.pred_seq_len))
        sigmas = enc_x.new_empty((self.batch_size, self.pred_seq_len))
        enc_h = self.encoder(enc_x)

        emb_t, h_t_1_tilde, h_t_1 = dec_emb[:,0,:], None, enc_h
        for t in range(1, self.pred_seq_len + 1):
            dec_y, dec_h = self.decoder(emb_t, h_t_1_tilde, h_t_1)
            # final outputs
            outputs[:, t-1], sigmas[:, t-1] = self.generateFromNormal(dec_y)
            
            if t == self.pred_seq_len:
                break
            # decoder input update
            emb_t = dec_emb[:,t,:]
            h_t_1_tilde = outputs[:, t-1]
            h_t_1 = dec_h
            
        return outputs, sigmas
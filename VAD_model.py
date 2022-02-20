import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from matplotlib import pyplot as plt


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def split_signal(signal, format='tensor', sample_rate=16000, duration=10):
    N = int(len(signal) / sample_rate * 1000 / duration)
    chunk_size = int(sample_rate * 10 / 1000)
    splitted_signal = [signal[i*chunk_size: (i+1)*chunk_size] for i in range(N)]
    if format == 'uint_16_bytes':
        to_16_bit = lambda x: torch.tensor(x * 2**15, dtype=torch.int16)
        splitted_signal = [to_16_bit(chunk).numpy().tobytes() for chunk in splitted_signal]
    return splitted_signal

class VADDataset(Dataset):
    def __init__(self, path, task_type='train', window_size=2):
        self.task_type = task_type
        self.window_size = window_size


        if task_type == 'train':            
            with open(path) as f:
                train_data = json.load(f)

            self.inputs = []
            self.labels = []
            for key, value in train_data.items():
                self.inputs.append(key)
                self.labels.append(torch.tensor(value, dtype=torch.long).flatten())
        else:
            self.inputs = [os.path.join(path, filename) for filename in  os.listdir(path)]

    def __getitem__(self, item):
        sample_rate = 16000
        input, _ = torchaudio.load(self.inputs[item])
        input = input.flatten()
        hmm_input = input
        
        splitted_signal = split_signal(input)
        webrtc_input = split_signal(input, 'uint_16_bytes')
        
        padding_chunk = torch.zeros(sample_rate // 100)
        padding = [padding_chunk for _ in range(self.window_size)]
        splitted_signal = padding + splitted_signal + padding
        chunks = []
        for idx in range(self.window_size, len(splitted_signal) - self.window_size):
            chunks.append(torch.cat(splitted_signal[idx - self.window_size: idx + (self.window_size + 1)]))
        batch = torch.stack(chunks)
        batch = torch.cat([
            batch,
            torch.view_as_real(torch.fft.fft(batch))[:, :, 0],
            torch.view_as_real(torch.fft.fft(batch))[:, :, 1]
        ], dim=-1)
        batch = torch.tensor(batch, dtype=torch.float)
        model_input = batch

        out = {'model_input': model_input}
        if self.task_type == 'train':
            out['label'] = self.labels[item]
        else:
            out['input_path'] = self.inputs[item]

        return out

    def __len__(self):
        return len(self.inputs)

class VAD_model(nn.Module):
    def __init__(self):
        super(VAD_model, self).__init__()
        self.hidden_dim = 256
        self.lstm = nn.LSTM(2400, 1024)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2)
        )

    def forward(self, batch):
        lstm_out, (hidden_states, _) = self.lstm(batch)
        input = torch.cat([lstm_out.squeeze(0), hidden_states.squeeze(0)], dim=-1)
        pred = self.classifier(input)
        return pred

def train(conf_file_path, num_epochs=3, device='cpu', weights_path=None):
    train_loader = DataLoader(VADDataset(conf_file_path), batch_size=1, shuffle=True) 

    loss_fn = nn.CrossEntropyLoss()
    rnn_vad = VAD_model().to(device)
    optimizer = optim.Adam(rnn_vad.parameters(), lr=0.03)


    if weights_path:
        rnn_vad.load_state_dict(torch.load(weights_path))

    if not os.path.exists('weights'):
        os.makedirs('weights')

    for epoch in range(num_epochs):
        for batch in tqdm(train_loader):
            rnn_vad.zero_grad()
            
            probs = rnn_vad(batch['model_input'].to(device))
            loss = loss_fn(probs, batch['label'].squeeze().to(device))
            loss.backward()
            optimizer.step()
            
        torch.save(rnn_vad.state_dict(), f'weights/{epoch}')

def evaluate(files_path, weights_path, device='cpu'):
    eval_loader = DataLoader(VADDataset(files_path, task_type='eval'), batch_size=1, shuffle=False)
    rnn_vad = VAD_model().to(device)
    rnn_vad.load_state_dict(torch.load(weights_path))
    rnn_vad.eval()

    preds_dct = dict()
    for batch in tqdm(eval_loader):
        with torch.no_grad():
            probs = rnn_vad(batch['model_input'].cuda())
            preds = torch.argmax(probs, dim=1)
            preds_dct[batch['input_path'][0]] = preds.detach().to('cpu')

    return preds_dct

def draw_mask(mask, amplitude):
    mask = torch.tensor(mask, dtype=torch.int16)
    ### upper side of plot
    plt.plot(torch.arange(len(mask)) * 160, mask * amplitude, color='lightgreen')
    plt.fill_between(torch.arange(len(mask)) * 160, amplitude, mask * amplitude, facecolor='red', alpha=0.3)
    plt.fill_between(torch.arange(len(mask)) * 160, 0, mask * amplitude, facecolor='lightgreen', alpha=0.3)
    ### lower side of plot
    plt.plot(torch.arange(len(mask)) * 160, -mask * amplitude, color='lightgreen')
    plt.fill_between(torch.arange(len(mask)) * 160, -amplitude, -mask * amplitude, facecolor='red', alpha=0.3)
    plt.fill_between(torch.arange(len(mask)) * 160, 0, -mask * amplitude, facecolor='lightgreen', alpha=0.3)

def draw_waveform(waveform):
    plt.plot(torch.arange(len(waveform.squeeze())), waveform.squeeze(), color='blue')

def draw_result(audio_file, mask):
    waveform, _ = torchaudio.load(audio_file)
    plt.rcParams["figure.figsize"] = (20, 12)
    draw_mask(mask, max(waveform.squeeze()))
    draw_waveform(waveform)
    plt.show()
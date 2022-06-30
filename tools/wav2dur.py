#!/usr/bin/env python3
# encoding: utf-8

import sys
import argparse
import json
import codecs
import yaml

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset, DataLoader

torchaudio.set_audio_backend("sox_io")

nj = 32
in_scp = sys.argv[1]
dur_scp = sys.argv[2]

class CollateFunc(object):
    ''' Collate function for AudioDataset
    '''
    def __init__(self, feat_dim=80, resample_rate=0):
        self.feat_dim = feat_dim
        self.resample_rate = resample_rate
        pass

    def __call__(self, batch):
        mean_stat = torch.zeros(self.feat_dim)
        var_stat = torch.zeros(self.feat_dim)
        number = 0
        keys = []
        shapes = []
        for item in batch:
            key = item[0]
            value = item[1].strip().split(",")
            assert len(value) == 3 or len(value) == 1
            wav_path = value[0]
            sample_rate = torchaudio.backend.sox_io_backend.info(wav_path).sample_rate
            resample_rate = sample_rate
            # len(value) == 3 means segmented wav.scp,
            # len(value) == 1 means original wav.scp
            if len(value) == 3:
                start_frame = int(float(value[1]) * sample_rate)
                end_frame = int(float(value[2]) * sample_rate)
                waveform, sample_rate = torchaudio.backend.sox_io_backend.load(
                    filepath=wav_path,
                    num_frames=end_frame - start_frame,
                    frame_offset=start_frame)
            else:
                waveform, sample_rate = torchaudio.load(item[1])

            waveform = waveform * (1 << 15)
            if self.resample_rate != 0 and self.resample_rate != sample_rate:
                resample_rate = self.resample_rate
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=resample_rate)(waveform)

            mat = kaldi.fbank(waveform,
                              num_mel_bins=self.feat_dim,
                              dither=0.0,
                              energy_floor=0.0,
                              sample_frequency=resample_rate)
            keys.append(key)
            shapes.append(mat.shape[0]/100.0)
        return keys, shapes


class AudioDataset(Dataset):
    def __init__(self, data_file):
        self.items = []
        with codecs.open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                arr = line.strip().split()
                self.items.append((arr[0], arr[1]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


if __name__ == '__main__':
    collate_func = CollateFunc()
    dataset = AudioDataset(in_scp)
    batch_size = 32
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             sampler=None,
                             num_workers=nj,
                             collate_fn=collate_func)
    fout = open(dur_scp, 'w')
    wav_number = 0
    utt2dur = {}
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            keys, numbers = batch
            for i in range(len(keys)):
                utt2dur[keys[i]] = numbers[i]
            wav_number += batch_size
            if wav_number % 1000 == 0:
                print(f'processed {wav_number} wavs', file=sys.stderr, flush=True)

    with codecs.open(in_scp, 'r', encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split()
            key = arr[0]
            fout.write('{} {}\n'.format(key, utt2dur[key]))

# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os
import sys
import k2
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import AudioDataset, CollateFunc
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.lexicon import Lexicon

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')

    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')


    args = parser.parse_args()
    print(args)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    raw_wav = configs['raw_wav']
    # Init dataset and data loader
    # Init dataset and data loader
    train_collate_conf = copy.deepcopy(configs['collate_conf'])
    train_collate_conf['spec_aug'] = False
    train_collate_conf['spec_sub'] = False
    train_collate_conf['feature_dither'] = False
    train_collate_conf['speed_perturb'] = False
    if raw_wav:
        train_collate_conf['wav_distortion_conf']['wav_distortion_rate'] = 0
        train_collate_conf['wav_distortion_conf']['wav_dither'] = 0.0
    train_collate_func = CollateFunc(**train_collate_conf, raw_wav=raw_wav)
    dataset_conf = configs.get('dataset_conf', {})
    dataset_conf['batch_size'] = args.batch_size
    dataset_conf['batch_type'] = 'static'
    dataset_conf['sort'] = False
    train_dataset = AudioDataset(args.train_data,
                                **dataset_conf,
                                raw_wav=raw_wav)
    train_data_loader = DataLoader(train_dataset,
                                  collate_fn=train_collate_func,
                                  shuffle=False,
                                  batch_size=1,
                                  num_workers=10)

    # Init asr model from configs
    model = init_asr_model(configs)

    # Load dict
    char_dict = {}
    with open(args.dict, 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
    eos = len(char_dict) - 1

    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)

    prior = torch.zeros(len(char_dict)).to(device)
    num_frames = 0.0
    model.eval()
    with torch.no_grad(), open(args.result_file, 'w') as fout:
        for batch_idx, batch in enumerate(train_data_loader):
            keys, feats, target, feats_lengths, target_lengths = batch
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)
            ctc_probs, encoder_out_lens = model.prior(
                    feats,
                    feats_lengths,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming)
            if batch_idx % 100 == 0:
                logging.debug(batch_idx)
            batch = ctc_probs.shape[0]
            for b in range(batch):
                ctc_prob = ctc_probs[b,:encoder_out_lens[b]]
                prior += torch.sum(ctc_prob, dim=0)
                num_frames += encoder_out_lens[b].item()
                
        prior = prior / num_frames
        prior_ = " ".join(['{:.15f}'.format(p.item()) for p in prior])
        fout.write( prior_ + '\n')

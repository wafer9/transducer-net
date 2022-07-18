# transducer-net

## rnnt + ctc + attention

* Feature info: using fbank feature, dither, cmvn, offline speed perturb
* Training info: lr 0.0003, batch size 16, 8 gpu, acc_grad 2, 100 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 60 (last 60)

| decoding mode             | CER   |
|---------------------------|-------|
| ctc greedy search         | 4.58 % N=104765 C=100061 S=4588 D=116 I=95   |
| attention rescoring       | 4.38 % N=104765 C=100264 S=4380 D=121 I=85   |
| tlg beam_size 20          | 4.36 % N=104765 C=100355 S=4270 D=140 I=159  |
| tlg + attention rescoring | 3.96 % N=104765 C=100737 S=3918 D=110 I=116  |
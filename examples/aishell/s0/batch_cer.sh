#!/bin/bash

dir=exp/conformer4
mode="ctc_fst_attention_rescore"
test_dir=$dir/test_${mode}
feat_dir=raw_wav

#cat $test_dir/text | awk '{print $2}' |sort |uniq >$test_dir/weight.list

#mkdir -p $test_dir/weight

while read line; do
  wid=$line
  grep $wid $test_dir/text | awk '{$2=""; print $0}' > $test_dir/weight/${wid}.txt
  python tools/compute-wer.py --char=1 --v=1 $feat_dir/test/text $test_dir/weight/${wid}.txt > $test_dir/weight/${wid}.cer

done < $test_dir/weight.list




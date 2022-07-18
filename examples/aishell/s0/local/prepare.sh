#!/usr/bin/env bash

set -eou pipefail
# . local/parse_options.sh || exit 1

stage=0
stop_stage=3
dl_dir=$1 # .*/aishell_set/

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  echo "stage -1: Download LM"
  # We assume that you have installed the git-lfs, if not, you could install it
  # using: `sudo apt-get install git-lfs && git-lfs install`
  if [ ! -f $dl_dir/lm/3-gram.unpruned.arpa ]; then
    git clone https://huggingface.co/pkufool/aishell_lm $dl_dir/lm
  fi
fi

lang_phone_dir=data/lang_phone
lang_char_dir=data/lang_char
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "Stage 0: Prepare phone based lang"
  mkdir -p $lang_phone_dir

  (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; ) |
    cat - $dl_dir/resource_aishell/lexicon.txt |
    sort | uniq > $lang_phone_dir/lexicon.txt

  ./local/generate_unique_lexicon.py --lang-dir $lang_phone_dir

  if [ ! -f $lang_phone_dir/L_disambig.pt ]; then
    ./local/prepare_lang.py --lang-dir $lang_phone_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Stage 1: Prepare char based lang"
  mkdir -p $lang_char_dir
  # We reuse words.txt from phone based lexicon
  # so that the two can share G.pt later.
  cp $lang_phone_dir/words.txt $lang_char_dir

  cat data/train/text |
  cut -d " " -f 2- | sed -e 's/[ \t\r\n]*//g' > $lang_char_dir/text

  if [ ! -f $lang_char_dir/L_disambig.pt ]; then
    ./local/prepare_char.py
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "Stage 2: Prepare G"
  # We assume you have install kaldilm, if not, please install
  # it using: pip install kaldilm

  mkdir -p data/lm
  if [ ! -f data/lm/G_3_gram.fst.txt ]; then
    # It is used in building HLG
    python3 -m kaldilm \
      --read-symbol-table="$lang_phone_dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      $dl_dir/lm/3-gram.unpruned.arpa > data/lm/G_3_gram.fst.txt
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "Stage 3: Compile HLG"
  #./local/compile_hlg.py --lang-dir $lang_phone_dir
  ./local/compile_hlg.py --lang-dir $lang_char_dir
fi


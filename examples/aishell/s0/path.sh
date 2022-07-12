KALDI_ROOT=/data/joe/asr/kaldi

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export WENET_DIR=$PWD/../../..
# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=.:$PYTHONPATH

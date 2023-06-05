#!/bin/bash
set -e

DEVICE=$1
EXP=$2
MODEL=$3

if grep -q "cnn" <<< "$EXP"; then
  CONFIG="cnndm"
elif grep -q "xsum" <<< "$EXP"; then
  CONFIG="xsum"
elif grep -q "samsum" <<< "$EXP"; then
  CONFIG="samsum"
else
  CONFIG="cnndm"
  echo "Dataset not in experiment name"
fi

MODEL_PT="${MODEL}/model_ranking.bin"
python main.py -e -r --cuda --gpuid $DEVICE --config $CONFIG --model_pt $MODEL_PT --experiment $EXP
python cal_rouge.py --ref ./result/${EXP}/reference_ranking --hyp ./result/${EXP}/candidate_ranking -l

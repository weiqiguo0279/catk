#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

DATA_SPLIT=validation # training, validation, testing

source ~/miniconda3/etc/profile.d/conda.sh
conda activate catk
python \
  -m src.data_preprocess \
  --split $DATA_SPLIT \
  --num_workers 12 \
  --input_dir /scratch/data/womd/uncompressed/scenario \
  --output_dir /scratch/cache/SMART
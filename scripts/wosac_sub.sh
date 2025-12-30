#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

ACTION=validate # validate, test
MY_EXPERIMENT="wosac_sub"
MY_TASK_NAME=$MY_EXPERIMENT-$ACTION"-debug"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate catk
python \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  action=$ACTION \
  task_name=$MY_TASK_NAME

# below is for training with ddp
# torchrun \
#   --rdzv_id $SLURM_JOB_ID \
#   --rdzv_backend c10d \
#   --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#   --nnodes $NUM_NODES \
#   --nproc_per_node gpu \
#   -m src.run \
#   experiment=$MY_EXPERIMENT \
#   trainer=ddp \
#   action=$ACTION \
#   task_name=$MY_TASK_NAME

echo bash $ACTION done!
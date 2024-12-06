#!/bin/bash

#SBATCH --job-name=local # job name
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks-per-node=1 # this is usually set to 1
#SBATCH --cpus-per-task=208 # same as above, number of cpus
#SBATCH --gres=gpu:8 # number of gpus
#SBATCH --time=1440:00:00 # runtime, 1440 is the max
#SBATCH --output=log/%j-%x/slurm.log # output log
#SBATCH --error=log/%j-%x/error.log # error log
#SBATCH --nodelist=a3mega-a3meganodeset-1

module load conda

source activate thai_local

TRANSFORMERS_CACHE="/shared/aisingapore/.cache"
HF_DATASETS_CACHE="/shared/aisingapore/.cache" 

bash eval_only.sh

# conda init

# conda deactivate
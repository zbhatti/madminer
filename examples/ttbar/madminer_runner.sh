#!/bin/bash

#SBATCH --job-name=zbhatti_madminer
#SBATCH --output=log_%a_%j.log
#SBATCH --nodes=1

## use for generate:
##SBATCH --mem=16GB
##SBATCH --time=2-23:00:00
##SBATCH --cpus-per-task=2

## use for train:
#SBATCH --mem=150GB
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4

# unused:
##SBATCH --gres=gpu:1
##SBATCH --mail-user=zb609@nyu.edu
##SBATCH --mail-type=END

export PATH=$PATH:$HOME/anaconda2/bin
source $HOME/anaconda2/etc/profile.d/conda.sh

conda activate python-mm-27
conda env list
source /scratch/zb609/root_build/bin/thisroot.sh

cd $SCRATCH/madminer/examples/ttbar
export MADGRAPH_DIR=/home/zb609/scratch_dir/MG5_aMC_v2_6_5

# RUN SETUP AS YOURSELF
# python -u batch_madminer_event_runner.py setup

# python -u batch_madminer_event_runner.py generate $1 ${SLURM_ARRAY_TASK_ID}
python -u batch_madminer_event_runner.py train

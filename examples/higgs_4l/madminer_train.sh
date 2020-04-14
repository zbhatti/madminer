#!/bin/bash

#SBATCH --job-name=zbhatti_madminer
#SBATCH --output=log_%a_%j.log
#SBATCH --nodes=1

## use for train:
#SBATCH --mem=150GB
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4

export PATH=$PATH:$HOME/anaconda2/bin
source $HOME/anaconda2/etc/profile.d/conda.sh

conda activate python-mm-27
conda env list
source /scratch/zb609/root_build/bin/thisroot.sh

cd $SCRATCH/madminer/examples/higgs_4l
export MADGRAPH_DIR=/home/zb609/scratch_dir/MG5_aMC_v2_6_5

python -u batch_madminer_event_runner.py $1 train

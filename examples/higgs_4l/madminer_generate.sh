#!/bin/bash

#SBATCH --job-name=zbhatti_madminer
#SBATCH --output=log_%a_%j.log
#SBATCH --nodes=1

## use for generate:
#SBATCH --mem=16GB
#SBATCH --time=4-23:00:00
#SBATCH --cpus-per-task=2

export PATH=$PATH:$HOME/anaconda2/bin
source $HOME/anaconda2/etc/profile.d/conda.sh

conda activate python-mm-27
conda env list
source /scratch/zb609/root_build/bin/thisroot.sh

cd $SCRATCH/madminer/examples/higgs_4l
export MADGRAPH_DIR=/home/zb609/scratch_dir/MG5_aMC_v2_6_5

# RUN SETUP AS YOURSELF
# python batch_madminer_event_runner.py ~/scratch_dir/madminer_data_6_dup setup

python -u batch_madminer_event_runner.py $1 generate $2 ${SLURM_ARRAY_TASK_ID}

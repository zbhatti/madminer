#!/bin/bash

#SBATCH --job-name=zbhatti_madminer
#SBATCH --output=log_%a_%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150GB
#SBATCH --time=20:00:00
##SBATCH --gres=gpu:1
##SBATCH --mail-user=zb609@nyu.edu
##SBATCH --mail-type=END

export PATH=$PATH:$HOME/anaconda2/bin
source $HOME/anaconda2/etc/profile.d/conda.sh

conda activate python-mm-27
conda env list
source /scratch/zb609/root_build/bin/thisroot.sh

cd $SCRATCH/madminer/examples/ttbar

#python -u batch_madminer_event_runner.py setup
#python -u batch_madminer_event_runner.py generate $1 ${SLURM_ARRAY_TASK_ID}
python -u batch_madminer_event_runner.py train

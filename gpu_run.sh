#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=nba_train
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:2

module purge
module load python/intel/2.7.12
module load pytorch/intel/20170125
module load scikit-learn/intel/0.18.1

cd /home/zz1409/Quora/siamese-net

python question_model.py

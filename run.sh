#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=zachzhang
#SBATCH --time=2:00:00


module purge
module load python3/intel/3.5.2
module load pytorch/intel/20170125
module load scikit-learn/intel/0.18.1

cd /home/zz1409/Quora/siamese-net

python prep_data_quora.py

#python siamese_net.py

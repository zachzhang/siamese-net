#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=190GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=zachzhang
#SBATCH --time=12:00:00


module purge
module load scikit-learn/intel/0.18.1
module load pytorch/intel/20170125

cd /home/zz1409/siamese-net

python artist_recognizer.py


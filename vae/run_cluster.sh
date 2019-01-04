#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
module load cuda/9.0.176

#module load tensorflow/1.4.0-py27-gpu
#module load cudnn/v5.1
#module load pytorch/0.2.0p3-py27
#module load pytorch/0.3.0-py27
#module load pytorch/0.4.0-py27
#module load keras/1.2.2

module load python/3.6.1

module load pytorch/0.4.1-py36-cuda90
#module load tensorflow/1.1.0-py35-cpu
module load torchvision/0.2.1-py36

#module load opencv/2.4.13.2
python3 main.py
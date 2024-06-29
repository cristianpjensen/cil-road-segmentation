#!/bin/bash
#SBATCH --account=cil
#SBATCH --gpus=1
#SBATCH --time=12:00:00

module purge
module load cuda/12.1

cd ~/cil-road-segmentation
source venv/bin/activate

python train.py

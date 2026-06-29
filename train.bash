#!/bin/bash
#SBATCH --job-name=spline_uncertainty
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=spline_%j.out

echo "Job started on:"
hostname
date

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dncnn

cd ~/structured_uncertainty

echo "Python used:"
which python

echo "Checking PyTorch:"
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"

python main.py

echo "Job finished:"
date

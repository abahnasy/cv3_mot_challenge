#!/bin/bash

#SBATCH -J electric_wires_dataset
#SBATCH -o /lustre/groups/imm01/workspace/ahmed.bahnasy/thesis-playground/09_slurm_logs/run_%j.out
#SBATCH -e /lustre/groups/imm01/workspace/ahmed.bahnasy/thesis-playground/09_slurm_logs/run_%j.err
#SBATCH -p gpu_p
#SBATCH -q gpu
#SBATCH -C gpu_tesla_v100
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -c 6
#SBATCH --mem=12G
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate mot_challenge
# cd /lustre/groups/imm01/workspace/ahmed.bahnasy/mot_challenge
python reid_main.py
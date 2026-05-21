#!/bin/bash
#SBATCH --job-name=slam_moon1_full
#SBATCH --output=slam_moon1_%j.out
#SBATCH --error=slam_moon1_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sc22hkd@leeds.ac.uk

module load miniforge
conda activate env

cd /mnt/scratch/sc22hkd/project/repo
pip install -e . --quiet
python analyse_moon1_full_hpc.py

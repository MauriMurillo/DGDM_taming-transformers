#!/bin/bash
#SBATCH --time=24:00:00              
#SBATCH --nodes=1            
#SBATCH --ntasks=1         
#SBATCH --mem=64GB              
#SBATCH --cpus-per-task=24    
#SBATCH --partition=gpu-a100 --gres=gpu:1
source /home/mauricio.murillo/.bashrc
conda activate taming_old
cd /home/mauricio.murillo/DGDM_taming-transformers
python main.py --base configs/classeg_vqgan_masks.yaml -n masks -t True --gpus 0, 
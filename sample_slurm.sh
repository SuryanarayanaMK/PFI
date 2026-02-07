#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -t 6:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=a100-sxm4-80gb:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# module load python/3.9.18
# module load cuda/12.3.2

source ~/.venvs/cfm311/bin/activate 
module load cuda/12.5.1

for dm in Langevin Mult Additive ODE; do
  for seed in 0 1 2 3 4; do
    python3 Train_on_exvivo_data.py \
      --fac 2 \
      --nmb 1 \
      --nsamples 6000 \
      --simflag False \
      --seed $seed \
      --Np 60 \
      --Nf 60 \
      --dm $dm \
      --spectral_flag True \
      --geneset_num 2 \
      --prescient_flag True
  done
done



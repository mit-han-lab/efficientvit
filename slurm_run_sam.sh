slurm_account=${slurm_account:-"YOUR_ACCOUNT"}
slurm_partition=${slurm_partition:-"YOUR_PARTITION"}

for i in $(seq 1 10); do 

export NNODES=4

srun -p $slurm_partition -N $NNODES -t 04:00:00 \
    -A $slurm_account -J efficientvit-sam-xl1 \
    --gpus-per-node 8 --exclusive \
    --dependency singleton \
    bash ~/efficientvit/train_sam_model.sh &

done
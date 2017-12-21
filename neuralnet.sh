#!/bin/sh
#SBATCH -p slurm_shortgpu
#SBATCH --time=0-00:05:00 # run time in days-hh:mm:ss
#SBATCH --nodes=1
#SBATCH -o job_out
#SBATCH -e job_err
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 # not needed for OpenMP
cd $SLURM_SUBMIT_DIR
#for i in {1 10 100 1000} ;
#do 
./finalProject 10 30000 10 .5 50
#done
mv job_out fpoffset.out
mv job_err fpoffset.err


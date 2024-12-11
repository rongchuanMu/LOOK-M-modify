#!/bin/bash
#SBATCH --job-name="train"
#SBATCH -o test.out
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 100:00:00                            
#SBATCH --cpus-per-task=16                                                               
#SBATCH -w gpu15
#SBATCH --gres=gpu:a100-pcie-40gb:1

cd
cd clash
./clash -d . &
export http_proxy="http://127.0.0.1:12397"
export HTTP_PROXY="http://127.0.0.1:12397"
export https_proxy="http://127.0.0.1:12397"
export HTTPS_PROXY="http://127.0.0.1:12397"
export all_proxy="socks5h://127.0.0.1:12382"
export ALL_PROXY="socks5h://127.0.0.1:12382"
cd
source ~/.bashrc
source ~/anaconda3/bin/activate
conda activate LOOK-M
cd /home/rcmu/read_papers/LOOK-M-main
pwd
bash ./scripts/origin_eval.sh
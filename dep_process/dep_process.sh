#!/bin/bash

exec >> ./dep_process.log 2>&1
time3=$(date "+%Y-%m-%d %H:%M:%S")
echo $time3

cd /home/chengkun/jupyter_projects/Magic-NLPer-main/re/dep_process/

MAX_LENGTH=100
current_datasets_path='/home/chengkun/jupyter_projects/Magic-NLPer-main/data/en_zh/'
parallel_corpus='news_en_zh_shuffle_final.txt'
ngpu=2
batch=120

python -u ./dataset_splite.py $MAX_LENGTH $current_datasets_path $parallel_corpus &
wait
echo $(date "+%Y-%m-%d %H:%M:%S")
python -u ./dep.py $MAX_LENGTH $current_datasets_path &
wait
echo $(date "+%Y-%m-%d %H:%M:%S")
python -u ./dep_save.py $ngpu $batch $MAX_LENGTH $current_datasets_path &
echo $(date "+%Y-%m-%d %H:%M:%S")

#tmux send -t dep_process "if [ $? == 0 ]; then echo Success >> ./dep_process.log; exit 0; else echo Fail >> ./dep_process.log; exit 0; fi" ENTER
#tmux send -t dep_process "$(echo -e "\n" >> ./log)" ENTER
#tmux kill-session -t dep_process
#tmux new-window "tmux detach"
#python3 -u ./1.py >> ./python.log && python3 -u ./2.py >> ./python.log && python3 -u ./3.py >> ./python.log && python3 -u ./4.py >> ./python.log && tmux detach








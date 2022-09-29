#!/bin/bash
path="/home/chengkun/jupyter_projects/Magic-NLPer-main/re/"
cd $path

exec >> ./pytask_256.log 2>&1
echo
time3=$(date "+%Y-%m-%d %H:%M:%S")
echo $time3

tmux kill-session -t pytask
if [ $? != 0 ]; then echo "can't find session pytask" ; else echo Session pytask closed ; fi

tmux new -s pytask -d #在后台建立会话
exit_code=$?
if [ $exit_code == 0 ]
then
	echo "Session pytask created"
else
	echo "Failed to create session pytask"
	exit 1
fi

tmux send -t pytask "conda activate pytorch" ENTER
tmux send -t pytask "exec >> ./pytask_256.log 2>&1" ENTER
tmux send -t pytask "cd $path" ENTER

gpu0=0
gpu1=2
#seed=1934 256
for seed in 256
do
	tmux send -t pytask "fuser -v /dev/nvidia{$[ $gpu0 + 2 ],$[ $gpu1 + 2 ]} |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | bash ; sleep 3;\
python -u ./transformer_improved_encoder_decoder_zero.py $gpu0 $gpu1 $seed || echo transformer_improved_encoder_decoder_zero.py failed; \
fuser -v /dev/nvidia{$[ $gpu0 + 2 ],$[ $gpu1 + 2 ]} |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | bash ; sleep 3;\
python -u ./transformer_improved_decoder.py $gpu0 $gpu1 $seed || echo transformer_improved_decoder.py failed ;\
fuser -v /dev/nvidia{$[ $gpu0 + 2 ],$[ $gpu1 + 2 ]} |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | bash ; sleep 3;\
python -u ./transformer_improved_encoder_decoder_dep.py $gpu0 $gpu1 $seed || echo transformer_improved_encoder_decoder_dep.py failed ;\
fuser -v /dev/nvidia{$[ $gpu0 + 2 ],$[ $gpu1 + 2 ]} |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | bash ; sleep 3;\
python -u ./transformer_improved_encoder.py $gpu0 $gpu1 $seed || echo transformer_improved_encoder.py failed ;\
fuser -v /dev/nvidia{$[ $gpu0 + 2 ],$[ $gpu1 + 2 ]} |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | bash ; sleep 3;\
python -u ./transformer_base.py $gpu0 $gpu1 $seed || echo transformer_base.py failed " ENTER
done


#tmux send -t pytask "if [ $? == 0 ]; then echo Success >> ./pytask.log; exit 0; else echo Fail >> ./pytask.log; exit 0; fi" ENTER

#tmux send -t pytask "$(echo -e "\n" >> ./log)" ENTER

#tmux kill-session -t pytask
#tmux new-window "tmux detach"
#python3 -u ./1.py >> ./python.log && python3 -u ./2.py >> ./python.log && python3 -u ./3.py >> ./python.log && python3 -u ./4.py >> ./python.log && tmux detach








FILE=pit.rev250.v1.35i.vs.45i_49i.mcts100.eps200.avg.log

cd /home/et/projects/alpha-nagibator
python checkers_pit_mp.py >> ../$FILE 2>&1
cd ..
echo $FILE > $FILE.info
cat $FILE | grep -E 'Total result|start at|end of match' >> $FILE.info
cat $FILE.info | mail -s "file" youraddress@gmail.com

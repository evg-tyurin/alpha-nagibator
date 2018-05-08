FILE=main.rev250.v1.59i.global.avg.mcts100.eps200.lr0.02.hist25.log

cd /home/et/projects/alpha-nagibator
for iter in 59 
do 
	python checkers_main.py $iter >> ../$FILE 2>&1
done
cd ..
echo $FILE > $FILE.info
cat $FILE | grep -E 'Total result' >> $FILE.info
cat $FILE.info | mail -s "file" youraddress@gmail.com

#!/bin/bash

if [ "$#" -ne "2" ]; then
	echo -e "\nUsage:\n\t$0 <in.gt_files_folder> <in.scripts_folder>\n"
	exit 1
fi

numQueries=$(ls $1/query_*rank*.txt | wc -l | xargs -I {} expr {} - 1)

for i in `seq 0 $numQueries`;
	do
	echo Computing precision recall curve: $1/query_$i.csv
	octave --silent --eval "addpath('$2');subplot(1,1,1);eval_performance('$1/query_$i.csv');print -dpng $1/query_$i.png;"
done

#!/bin/bash
# replaces all "+" with "/" 

for f in *.jpg; do
	# replace all (//) occurences of + (+) with / (\/)
	unflattened="${f//+/\/}"
	# echo $unflattened
	mkdir -p `dirname $unflattened`
	mv $f $unflattened

done

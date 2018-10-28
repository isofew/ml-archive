#! /bin/sh

while true
do
	while read line
	do
		$line
		echo
	done <$1 >$2
done

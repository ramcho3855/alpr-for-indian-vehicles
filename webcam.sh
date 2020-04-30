#!/bin/bash


check_dir() 
{
	if [ ! -d "$1" ]
	then
		return 0
	else
		return 1
	fi
}


# Check if Darknet is compiled
check_file "darknet/libdarknet.so"
retval=$?
if [ $retval -eq 0 ]
then
	echo "Darknet is not compiled! Go to 'darknet' directory and 'make'!"
	exit 1
fi

tmp_dir='temp/frames'
tmp_output_dir='temp/output'


# Check if temp dir exists, if not, create it
check_dir $tmp_dir
retval=$?
if [ $retval -eq 0 ]
then
	mkdir -p $tmp_dir
fi



# Check if temp output dir exists, if not, create it
check_dir $tmp_output_dir
retval=$?
if [ $retval -eq 0 ]
then
	mkdir -p $tmp_output_dir
fi


# End if any error occur
set -e

# Do recognition
python3 webcam.py $tmp_dir $tmp_output_dir 




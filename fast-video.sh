#!/bin/bash

check_file() 
{
	if [ ! -f "$1" ]
	then
		return 0
	else
		return 1
	fi
}

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
input_path=''
tmp_output_dir='temp/output'
csv_file=''




# Check # of arguments
usage() {
	echo ""
	echo " Usage:"
	echo ""
	echo "   bash $0 -i input/dir -o output/dir [-h] :"
	echo ""
	echo "   -i   Input video path | 0 for webcam"
	echo "   -h   Print this help information"
	echo ""
	exit 1
}

while getopts 'i:o:h' OPTION; do
	case $OPTION in
		i) input_path=$OPTARG;;
		o) output_dir=$OPTARG;;
		h) usage;;
	esac
done


if [ -z "$input_path"  ]; then echo "Input path not set."; usage; exit 1; fi



# Check if temp dir exists, if not, create it
check_dir $tmp_dir
retval=$?
if [ $retval -eq 0 ]
then
	mkdir -p $tmp_dir
fi


# Check if input file exist
if [ $input_path -eq '0' ]
then
	echo "Reading webcam..."
else
check_file $input_path
retval=$?
if [ $retval -eq 0 ]
then
	echo "Input file ($input_dir) does not exist"
	exit 1
fi
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
python3 fast-video.py $tmp_dir $input_path $tmp_output_dir 




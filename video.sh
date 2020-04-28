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
output_dir=''
csv_file=''




# Check # of arguments
usage() {
	echo ""
	echo " Usage:"
	echo ""
	echo "   bash $0 -i input/dir -o output/dir -c csv_file.csv [-h] :"
	echo ""
	echo "   -i   Input video path"
	echo "   -o   Output dir path"
	echo "   -c   Output CSV file path"
	echo "   -h   Print this help information"
	echo ""
	exit 1
}

while getopts 'i:o:c:h' OPTION; do
	case $OPTION in
		i) input_path=$OPTARG;;
		o) output_dir=$OPTARG;;
		c) csv_file=$OPTARG;;
		h) usage;;
	esac
done


if [ -z "$input_path"  ]; then echo "Input path not set."; usage; exit 1; fi
if [ -z "$output_dir" ]; then echo "Ouput dir not set."; usage; exit 1; fi
if [ -z "$csv_file"   ]; then echo "CSV file not set." ; usage; exit 1; fi


# Check if temp dir exists, if not, create it
check_dir $tmp_dir
retval=$?
if [ $retval -eq 0 ]
then
	mkdir -p $tmp_dir
fi


# Check if input file exist
check_file $input_path
retval=$?
if [ $retval -eq 0 ]
then
	echo "Input file ($input_dir) does not exist"
	exit 1
fi

# Check if output dir exists, if not, create it
check_dir $output_dir
retval=$?
if [ $retval -eq 0 ]
then
	mkdir -p $output_dir
fi

# End if any error occur
set -e

# Do recognition
python video.py $tmp_dir $input_path $output_dir $csv_file 




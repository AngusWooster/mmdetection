#!/usr/bin/bash

source_path=$1/images
output_path=$1

output_file_name='annotation.txt'

if [ -z "$source_path" ] || [ -z "$output_path" ]
    then
    echo "no argument"
    echo "please assign \"source_path\" and \"output_path\""
    echo "ex: ${0} dataset/images dataset/annotation"
    exit 1
fi

# echo "ls ${source_path} -1 | grep xml | sed -e 's/.xml//'"
ls ${source_path} -1 | grep xml | sed -e 's/.xml//' > ${output_path}/${output_file_name}
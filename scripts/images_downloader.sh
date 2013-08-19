#!/bin/bash

if [ "$#" -ne "2" ]; then
	echo "Usage: $0 <input.geotag> <output.folder.prefix>"
	exit 1
fi

export DATASET_METADATA_FOLDER=$HOME/mediaeval_dataset/metadata
export DATASET_COORDINATES_FOLDER=$HOME/mediaeval_dataset/coordinates
export DATASET_TEST_FOLDER=$HOME/mediaeval_dataset/test

mkdir -p $DATASET_METADATA_FOLDER/../images
export DATASET_IMAGES_FOLDER=$HOME/mediaeval_dataset/images

# Count how many images contain the specified tag
ls $DATASET_METADATA_FOLDER/metadata*1*.gz | xargs -I {} zcat {} | grep -i $1 | wc -l | xargs -I {} echo "Found ["{}"] images containing the tag [$1]"

# Download the images containing the specified tag
echo "Proceeding to download them"

mkdir -p $DATASET_IMAGES_FOLDER/../$2
ls $DATASET_METADATA_FOLDER/metadata*1*.gz | xargs -I {} zcat {} | grep -i $1 | cut -d"," -s -f4 | xargs -I {} wget -nc -nv -P $DATASET_IMAGES_FOLDER/../$2 {}


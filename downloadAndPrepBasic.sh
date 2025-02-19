#!/bin/bash

#downlaod data
wget -O DATASETS.tar.xz 'https://www.dropbox.com/s/qtowh6tq57kd2ss/DATASETS.tar.xz?dl=1'
tar xvf DATASETS.tar.xz
rm DATASETS.tar.xz

mkdir images
mkdir test_labels
mkdir train_labels

# doing to use a min set of images to start
cp DATASETS/voc/test/images/* images
cp DATASETS/voc/train/images/* images

cp DATASETS/voc/test/annotations/* test_labels
cp DATASETS/voc/train/annotations/* train_labels

rm -fR DATASETS
ls -li
pwd
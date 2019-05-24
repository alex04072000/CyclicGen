#! /bin/bash
set -e

# Extract frame triplets from the UCF-101 dataset
# (http://crcv.ucf.edu/data/UCF101.php). Run as follows:
#
#   ./extract-ucf101.sh dir/file.avi
#
# Or, with parallel from moreutils, you can do it for all videos
# over many cores:
#
#   parallel -j 12 ./extract-ucf101.sh ::: $( find -name \*.avi )
#   //parallel -j 12 ./extract-ucf101.sh -- $( find -name \*.avi )
#
# but do note that this will produce ~250 GB of PNGs, probably many
# more frames than you actually would get to use for training
# and likely straining the file system with ~5M files.
#
# The script will create a set of frame files that you can easily combine
# and use for training:
#
#  for N in 1 2 3; do echo $N; cat $( find -name \*_frame$N.txt | sort -u )  > ../frame$N.txt; done

FILE=$1
PREFIX=$( dirname $FILE )/$( basename $FILE | tr -c "a-zA-Z0-9_-" "_" )

echo $FILE

ffmpeg -loglevel error -i $FILE -vf scale=256:256 ${PREFIX}_%04d.png

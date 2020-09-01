#!/bin/bash
# call this script from .. relative to this file's location.
rendermode=svg
colormap=Accent
seed=0
numsamples=5
logdir=./output-toy-figure-mk2
logfile=$logdir/log.txt

#rm $logdir/*.svg
#rm $logfile

for dataset in circle moon mult ; do
    for criterion in grad lrp taylor weight ; do
        python main.py  --rendermode $rendermode \
                        --colormap $colormap \
                        --logfile $logfile \
                        --dataset $dataset \
                        --criterion $criterion \
                        --numsamples $numsamples \
                        --seed $seed
    done
done
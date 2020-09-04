#!/bin/bash
# call this script from .. relative to this file's location.
rendermode=svg
colormap=Accent
seed=0
numsamples=5 # has no effect on rendering the with --noisytest test data


for noiselevel in 0.01 0.1 0.3 ; do
    logdir=./output-toy-figure-mk2-noisytest/noise-$noiselevel
    mkdir $logdir
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
                            --seed $seed \
                            --noisytest $noiselevel
        done
    done
done
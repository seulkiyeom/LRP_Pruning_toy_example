#!/bin/bash
for s in *-*.sh;
do
    sed -i 's/rankanalysis/rankanalysis-revision2/g' $s
done

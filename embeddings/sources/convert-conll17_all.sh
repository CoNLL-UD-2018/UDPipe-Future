#!/bin/sh

while read line; do
  qsub -q cpu-ms.q@* -j y bash convert-conll17.sh $line
done <convert-conll17.langs

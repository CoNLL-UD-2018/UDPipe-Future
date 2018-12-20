#!/bin/sh

modeldir=$1; shift
outdir=$1; shift

mkdir -p $outdir
while read lang; do
  qsub -q cpu-*@* -pe smp 8 -l mem_free=8G -o $outdir/$lang.eval -e $outdir/$lang.log sh test.sh $lang $modeldir
done <langs.big

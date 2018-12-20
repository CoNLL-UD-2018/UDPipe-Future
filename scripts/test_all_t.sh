#!/bin/sh

modeldir=$1; shift
outdir=$1; shift
threads=$1; shift

mkdir -p $outdir
while read lang; do
  qsub -q cpu-troja.q@* -pe smp 16 -l mem_free=8G -o $outdir/$lang.eval -e $outdir/$lang.log sh test_t.sh $lang $modeldir $threads
done <langs.big

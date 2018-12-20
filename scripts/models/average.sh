#!/bin/sh

lang="$1"; shift
dir="$1"; shift
script="$1"; shift

mkdir -p $dir
for d in ../logs/$dir.py*b=$lang,*; do
  [ -d "$d" ] || continue
  id=$(perl -ple 's/^\S*\s+//;s/=embeddings\//=/g;s/-?-?([a-z])[a-z]*_?/$1/g;s/\s/,/g' $d/cmd)
#  for avg in best1 best12 best123 last1 last12 last123 last12345; do # last1234567 last156 last167; do
#  for avg in best1 last1 last12 last123 last12345; do
  for avg in last1; do
    target=$dir/$lang-$id-$avg
    checkpoints=$(ls -v $d/checkpoint-inference-${avg%%[0-9]*}*.index | perl -nle '
    s/\.index$//;
    push @f, $_;
    END {
      @s=split //,"'${avg##*[a-z]}'";
      foreach (@s) { exit(1) if $_ > @f }
      print join(";", map {$f[-$_]} @s)
    }')
    [ -z "$checkpoints" ] && continue

    (echo -n ../../venv-cpu/bin/python3 ../../$script --predict --predict_input=/dev/stdin --predict_output=/dev/stdout --checkpoint=${target##*/}.model --threads=8 ../../; cat $d/cmd) >$target.cmd
    echo $target
    qsub -q cpu-ms.q@* -j y -o /dev/null ../venv-cpu/bin/python3 ../$script --predict --predict_input=/dev/null --predict_output=/dev/null --predict_save_checkpoint=$target.model --checkpoint="'$checkpoints'" ../$(cat $d/cmd)
  done
done

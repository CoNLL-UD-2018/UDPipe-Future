#!/bin/sh

filter="$1"; shift
outdir="$1"; shift

mkdir -p ../$outdir
for lang in $(cat ../langs); do
  # If there is only one cmd matching filter, use it; otherwise choose the best
  files=$(ls */$lang-*.cmd | sed 's/.cmd$/ /' | grep -e "$filter" | wc -l)
  if [ $files -eq 1 ]; then
    best=$(ls */$lang-*.cmd | sed 's/.cmd$/ /' | grep -e "$filter" | head -n1 | awk '{print $1}')
  else
    best=$(./results.sh $lang | grep -e "$filter" | head -n1 | awk '{print $1}')
  fi
  echo For $lang chosen $best
  for f in $best.*; do
    outf=../$outdir/$(basename $f)
    case $f in
      *.cmd) sed 's#\(^\| \)../#\1#g; s#^udpipe#./udpipe#; s#--embeddings=#--embeddings=../#' $f >$outf;;
      *) ln -s ../models/$f $outf
    esac
  done
done

#!/bin/sh

outdir=$1; shift

cat $outdir/*.log | perl -ne '
  /Prediction of (\d+) words took ([\d.]+) wall time/ or next;
  $words += $1;
  $time += $2;
  $total += 1;
  printf "%.3f\n", $words / $time;

  END {
    printf STDERR "There was %d words in %d treebanks, total time %.3f, on average %.1f w/s\n", $words, $total, $time, $words / $time;
  }
'

#!/bin/sh

outdir=$1; shift

head -qn1 $outdir/*.eval | perl -ne '
  @nums = split /[^0-9.]+/;
  if (!@sums) {
    @sums = @nums;
    @texts = split /[0-9.]+/;
  } else {
    for (my $i = 0; $i < @nums; $i++) {
      $sums[$i] += $nums[$i];
    }
  }
  $total += 1;

  END {
    for (my $i = 0; $i < @sums; $i++) {
      printf "%s%.3f", $texts[$i], $sums[$i] / $total;
    }
    print " $total\n";
  }
' >$outdir/EVAL

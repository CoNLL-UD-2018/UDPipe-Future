#!/bin/sh

lang=$1; shift
modeldir=$1; shift
threads=$1; shift

tokenizer-chosen/udpipe --tokenizer=normalized_spaces tokenizer-chosen/$lang.model ud-2.2/$lang/$lang-ud-test.txt \
  | (cd $modeldir && $(cat $lang-*.cmd | sed "s/threads=8/threads=$threads/") --predict --predict_input=/dev/stdin --predict_output=/dev/stdout) \
  | python3 conll18_ud_eval.py -v ud-2.2/$lang/$lang-ud-test.conllu /dev/stdin | perl -ne '
  $c .= $_;
  @p = split /\s+/;
  @p == 9 and $r{$p[0]} = $p[6];
  END {
    $a = ($r{"LAS"} + $r{"MLAS"} + $r{"BLEX"}) / 3;
    print "$a " . join(" ", map {"$_=$r{$_}"} qw(UAS LAS MLAS BLEX UPOS XPOS UFeats AllTags Lemmas)) . "\n";
    print $c;
  }
'

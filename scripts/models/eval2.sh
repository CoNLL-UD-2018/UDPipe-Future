#!/bin/sh

lang="$1"; shift
for cmd in "$@"; do
  ./udpipe --tokenizer=normalized_spaces tokenizer-best/$lang.model ../ud-2.2-conll18/$lang/$lang-ud-dev.txt \
    | (cd $(dirname $cmd) && $(cat $(basename $cmd))) \
    | python3 ../conll18_ud_eval.py -v ../ud-2.2-conll18/$lang/$lang-ud-dev.conllu /dev/stdin \
    | perl -ne '
  $c .= $_;
  @p = split /\s+/;
  @p == 9 and $r{$p[0]} = $p[6];
  END {
    $a = ($r{"LAS"} + $r{"MLAS"} + $r{"BLEX"}) / 3;
    print "$a " . join(" ", map {"$_=$r{$_}"} qw(UAS LAS MLAS BLEX UPOS XPOS UFeats AllTags Lemmas)) . "\n";
    print $c;
  }' | head -n1 #>${cmd%.cmd}.eval #tok-ori.eval
done

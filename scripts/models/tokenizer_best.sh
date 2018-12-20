#!/bin/bash

for d in tokenizer-ori/*.dev; do
  for m in $d tokenizer-new/${d#tokenizer-ori/}; do
    echo -n "${m%.dev} "
    perl -ne '
    /^Tokenizer (?:words|sentences).*precision: ([0-9.]*%).*recall: ([0-9.]*%).*f1: ([0-9.]*%)/ && print "$1P/$2R/$3 ";
    ' $m
    echo
  done | perl -aF'/\s|%[PR]\//' -ne 'push @l,$_; $l{$_}=2*$F[3]+$F[6]; END{foreach (sort {$l{$b}<=>$l{$a}} @l) {print}}' \
       | tee /dev/stderr \
       | head -n1 \
       | while read what score; do
    true
    #cp $what.* tokenizer-best/
    #echo $what >tokenizer-best/${what#*/}.source
  done
done

#!/bin/sh

while read code; do
  best=0
  args=""
  for d in logs/ud_parser4.py-*b=$code,*/; do
    value=$(tail -n1 $d/log | perl -nle '
      @p = split(/[ ,:]+/);
      $v = ($p[18] + $p[22] + $p[24]) * 100;
      print int($v);
    ')
    if [ $value -gt $best ]; then
      best=$value
      args=$(cut -d" " -f2- $d/cmd)
    fi
  done

  echo $code $args
done <langs

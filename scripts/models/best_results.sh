#!/bin/sh

filter="$1"; shift

for lang in $(cat ../langs); do
  best=$(./results.sh $lang | grep -e "$filter" | head -n1 | awk '{print $1, $2}')
  echo $lang $best >&2
  echo ${best##* }
done

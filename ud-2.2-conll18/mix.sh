#!/bin/sh

mkdir -p mix
>mix/mix-ud-train.conllu
>mix/mix-ud-dev.conllu

for d in */; do
  d=${d%/}
  [ $d = mix ] && continue

  echo $d >&2
  awk '{print}; /^$/{if (i++ >= 200) exit}' $d/$d-ud-train.conllu >>mix/mix-ud-train.conllu
  awk '{print}; /^$/{if (i++ >= 20) exit}' $d/$d-ud-dev.conllu >>mix/mix-ud-dev.conllu

  for conllu in "$code"/*.conllu; do
    perl conllu_to_text.pl --language="$code" <"$conllu" >"${conllu%.conllu}.txt"
  done
done

perl conllu_to_text.pl --language=en <mix/mix-ud-train.conllu >mix/mix-ud-train.txt
perl conllu_to_text.pl --language=en <mix/mix-ud-dev.conllu >mix/mix-ud-dev.txt

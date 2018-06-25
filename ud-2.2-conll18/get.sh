#!/bin/sh

wget http://ufal.mff.cuni.cz/~zeman/soubory/release-2.2-st-train-dev-data.zip
unzip release-2.2-st-train-dev-data.zip
rm release-2.2-st-train-dev-data.zip
for tb in release-2.2-st-train-dev-data/ud-treebanks-v2.2/*/*-ud-train.conllu; do
  dir=`dirname $tb`
  long_name=`basename $dir`
  code=`basename $tb`
  code=${code%%-*}
  echo $code $long_name | tee -a iso_names.txt

  mkdir -p "$code"
  if [ -f "$dir"/"$code"-ud-dev.conllu ]; then
    cp "$dir"/"$code"-ud-train.conllu "$dir"/"$code"-ud-dev.conllu "$code"
  else
    perl conllu_split.pl "$code" "$code" <"$dir"/"$code"-ud-train.conllu
  fi

  for conllu in "$code"/*.conllu; do
    perl conllu_to_text.pl --language="$code" <"$conllu" >"${conllu%.conllu}.txt"
  done
done
rmdir release-2.2-st-train-dev-data/

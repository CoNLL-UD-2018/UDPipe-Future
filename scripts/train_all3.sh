#!/bin/sh

while read code; do
  case $code in
    cs_pdt|ru_syntagrus|cs_cac|hi_hdtb|ar_padt|es_ancora|ca_ancora|fr_gsd) continue;;
  esac
  best=0
  args=""
  args_other=""
  for d in logs/ud_parser3.py-*b=$code,*/; do
    value=$(tail -n1 $d/log | perl -nle '
      @p = split(/[ ,:]+/);
      $v = ($p[18] + $p[22] + $p[24]) * 100;
      print int($v);
    ')
    if [ $value -gt $best ]; then
      best=$value
      args_other="$args"
      args=$(cut -d" " -f2- $d/cmd)
    else
      args_other=$(cut -d" " -f2- $d/cmd)
    fi
  done

  for embeddings in embeddings/$code.embeddings.npy embeddings/$code.conll17.embeddings.npy; do
    [ -f $embeddings ] && qsub -q gpu*@dll* -cwd -b y -l gpu=1,gpu_cc_min3.5=1 -j y withcuda venv/bin/python3 ud_parser4.py ud-2.2-conll18/$code/$code $args_other --embeddings=${embeddings%.embeddings.npy}
  done
done <langs

#!/bin/sh

for code in mix; do
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

  if [ -f embeddings/$code.embeddings.npy -o -f embeddings/$code.conll17.embeddings.npy ]; then
    true
  else
    qsub -q gpu*@dll* -cwd -b y -p -99 -l gpu=1,gpu_cc_min3.5=1 -j y withcuda venv/bin/python3 ud_parser3b.py ud-2.2-conll18/$code/$code $args --we_dim=0
    qsub -q gpu*@dll* -cwd -b y -p -99 -l gpu=1,gpu_cc_min3.5=1 -j y withcuda venv/bin/python3 ud_parser3b.py ud-2.2-conll18/$code/$code $args_other --we_dim=0
  fi
done

#!/bin/bash

[ "$#" -ge 2 ] || { echo Usage: $0 input_dataset output_directory >&2; exit 1; }

input="$1"
output="$2"
workdir=/home/UDPipe-Future/conll2018_udpipe

cat "$input"/metadata.json | python3 -c "
import sys,json
for entry in json.load(sys.stdin):
    print(' '.join([entry['lcode'], entry['tcode'], entry['rawfile'], entry['outfile']]))
" | while read lcode tcode in out; do
      code=${lcode}_${tcode}
      case $code in
        br_keb) model=mix;;
        cs_pud) model=cs_pdt;;
        en_pud) model=en_ewt;;
        fo_oft) model=mix;;
        fi_pud) model=fi_tdt;;
        ja_modern) model=ja_gsd;;
        pcm_nsc) model=mix;;
        sv_pud) model=sv_talbanken;;
        th_pud) model=mix;;
        *) model=$code;;
      esac
      cmd=$workdir/models-chosen/$model-*.cmd

      echo Processing $code with model for code $model and cmd $cmd
      time ($workdir/tokenizer-chosen/udpipe --tokenizer=normalized_spaces $workdir/tokenizer-chosen/$model.model $input/$in \
            | (cd $(dirname $cmd) && $(cat $(basename $cmd))) > $output/$out)
    done

echo All done

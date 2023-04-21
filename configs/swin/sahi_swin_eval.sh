#!/bin/bash
set -v
BaseName=$(basename $BASH_SOURCE)
outname=${BaseName:0:-3}
outdir=runs/$outname
if [ ! -d "$outdir" ];then
mkdir -p $outdir
fi
cp $BASH_SOURCE $outdir

python -u main.py \
    --output_dir $outdir \
    --backbone swin \
    --img_size 448 \
    --coco_path $(dirname $(dirname "$PWD"))/data-1/ \
    --sahi \
    --resume ${outdir:0:-5}/checkpoint.pth \
    --eval \
2>&1 | tee -a $outdir/$outname.log
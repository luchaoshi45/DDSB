#!/bin/sh
set -v
BaseName=$(basename $BASH_SOURCE)
outname=${BaseName:0:-3}
outdir=runs/$outname
if [ ! -d "$outdir" ];then
mkdir -p $outdir
fi
cp $BASH_SOURCE $outdir

python -u main.py \
    --lr 2e-4 \
    --batch_size 7 \
    --epochs 100 \
    --milestones 70 95 \
    --save_pth_fre 20 \
    --output_dir $outdir \
    --backbone resnet50 \
    --img_size 896 \
    --coco_path $(dirname $(dirname "$PWD"))/data-1/ \
    --cam \
2>&1 | tee -a $outdir/$outname.log
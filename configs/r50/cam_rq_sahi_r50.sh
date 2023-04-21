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
    --batch_size 25 \
    --epochs 35 \
    --milestones 20 30 \
    --save_pth_fre 10 \
    --output_dir $outdir \
    --backbone resnet50 \
    --img_size 448 \
    --coco_path $(dirname $(dirname "$PWD"))/data-2/ \
    --res_query \
    --cam \
2>&1 | tee -a $outdir/$outname.log
#!/bin/bash
set -x
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd $SCRIPT_DIR/

if [ -z "$(pip list | grep pretrain3d)" ]; then
    pip install -e . --user
fi

cuda=0
POSITIONAL=()
pretrainedmodel=/pretrainedmodel

while [[ $# -gt 0 ]]; do
    key=$1
    case $key in
    -m)
        pretrainedmodel=$2
        shift 2
        ;;
    -c | --cuda)
        cuda=$2
        shift 2
        ;;
    *)
        POSITIONAL+=("$1")
        shift
        ;;
    esac
done

SAVEDIR=/tmp/tmpckt
mkdir -p $SAVEDIR/
CUDA_VISIBLE_DEVICES=$cuda python finetune.py \
        --checkpoint-dir $SAVEDIR ${POSITIONAL[@]} \
        --finetune-from $pretrainedmodel | tee $SAVEDIR/training.log

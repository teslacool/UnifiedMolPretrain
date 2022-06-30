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
dist=false
prefix=pretrain3d
port=29500
restore=false

while [[ $# -gt 0 ]]; do
    key=$1
    case $key in
    -c | --cuda)
        cuda=$2
        shift 2
        ;;
    --dist)
        dist=true
        shift
        ;;
    --prefix)
        prefix=$2
        shift 2
        ;;
    --port)
        port=$2
        shift 2
        ;;
    --restore)
        restore=true
        shift
        ;; 
    *)
        POSITIONAL+=("$1")
        shift
        ;;
    esac
done
SUFFIX=$(echo ${POSITIONAL[*]} | sed -r 's/-//g' | sed -r 's/\s+/-/g')
if [ "$restore" == "true" ]; then 
    POSITIONAL+=("--restore")
fi 
SAVEDIR=/model/pretrain3d/$prefix
if [ -n "$SUFFIX" ]; then
    SAVEDIR=${SAVEDIR}-${SUFFIX}
fi
mkdir -p $SAVEDIR
if [ "$dist" == true ]; then
    cudaa=$(echo $cuda | sed -r 's/,//g')
    nproc=${#cudaa}
    CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --master_port $port --nproc_per_node=$nproc train.py \
        --checkpoint-dir $SAVEDIR ${POSITIONAL[@]} | tee $SAVEDIR/training.log
else
    CUDA_VISIBLE_DEVICES=$cuda python train.py \
        --checkpoint-dir $SAVEDIR ${POSITIONAL[@]} | tee $SAVEDIR/training.log
fi

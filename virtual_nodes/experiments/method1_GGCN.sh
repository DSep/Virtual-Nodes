#!/bin/bash

set -e

# Variables
declare -a directed=("--directed" " ")
declare -a embedding_layer=("--use_embed" " ")
GENERIC_PARAM="--splits 3 --n_seeds 3 --epochs 1500 --patience 100 --verbosity 0 --clip --augment --augment_ratio 0.15 --include_vnode_labels"
LAYER_2="--layer 2"

# Operations
cd Heterophily_and_oversmoothing


for d in "${directed[@]}"
do
  for emb in "${embedding_layer[@]}"
  do
    echo "EXPERIMENT SET: ${d} ${emb}"

    # GCN
    python -u full-supervised.py --data wisconsin   $GENERIC_PARAM --layer 5  --weight_decay 6e-4  --model GGCN --hidden 16 --dropout 0
    python -u full-supervised.py --data texas       $GENERIC_PARAM --layer 2  --weight_decay 1e-3  --model GGCN --hidden 16 --dropout 0.4
    python -u full-supervised.py --data cora        $GENERIC_PARAM --layer 32 --weight_decay 1e-3  --model GGCN --hidden 16 --decay_rate 0.9
    python -u full-supervised.py --data citeseer    $GENERIC_PARAM --layer 10 --weight_decay 1e-7  --model GGCN --hidden 80 --decay_rate 0.02
    python -u full-supervised.py --data chameleon   $GENERIC_PARAM --layer 5  --weight_decay 1e-2  --model GGCN --hidden 32 --dropout 0.3 --decay_rate 0.8
    python -u full-supervised.py --data cornell     $GENERIC_PARAM --layer 6  --weight_decay 1e-3  --model GGCN --hidden 16 --decay_rate 0.7
    python -u full-supervised.py --data squirrel    $GENERIC_PARAM --layer 2  --weight_decay 4e-3  --model GGCN --hidden 32 --dropout 0.5
    python -u full-supervised.py --data film        $GENERIC_PARAM --layer 4  --weight_decay 7e-3  --model GGCN --hidden 32 --dropout 0 --decay_rate 1.2
    python -u full-supervised.py --data pubmed      $GENERIC_PARAM --layer 5  --weight_decay 1e-5  --model GGCN --hidden 32 --use_sparse --dropout 0.3 --decay_rate 1.1
  done
done #  > results.txt


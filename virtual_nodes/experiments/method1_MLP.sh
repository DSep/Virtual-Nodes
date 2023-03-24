#!/bin/bash

set -e

# Variables
declare -a khops=("--khops 1" "--khops 2")
GENERIC_PARAM="--splits 3 --n_seeds 3 --epochs 1500 --patience 100 --verbosity 0 --augment --augment_ratio 1.0 --include_vnode_labels --clip"
LAYER_2="--layer 2"

# Operations
cd Heterophily_and_oversmoothing

for k in "${khops[@]}"
do
  echo "EXPERIMENT SET: ${GENERIC_PARAM} ${k} ${LAYER_2}"
  # MLP 
  python -u full-supervised.py --data cornell     $GENERIC_PARAM $k $LAYER_2 --weight_decay 1e-4 --model MLP --hidden 16               
  python -u full-supervised.py --data texas       $GENERIC_PARAM $k $LAYER_2 --weight_decay 1e-4 --model MLP --hidden 16               
  python -u full-supervised.py --data wisconsin   $GENERIC_PARAM $k $LAYER_2 --weight_decay 1e-4 --model MLP --hidden 16               
  python -u full-supervised.py --data chameleon   $GENERIC_PARAM $k $LAYER_2 --weight_decay 1e-8 --model MLP --hidden 16               
  python -u full-supervised.py --data cora        $GENERIC_PARAM $k $LAYER_2 --weight_decay 1e-4 --model MLP --hidden 16               
  python -u full-supervised.py --data citeseer    $GENERIC_PARAM $k $LAYER_2 --weight_decay 1e-7 --model MLP --hidden 64               
  python -u full-supervised.py --data film        $GENERIC_PARAM $k $LAYER_2 --weight_decay 1e-3 --model MLP --hidden 16 --dropout 0   
  python -u full-supervised.py --data squirrel    $GENERIC_PARAM $k $LAYER_2 --weight_decay 1e-4 --model MLP --hidden 16 --dropout 0.6 
  python -u full-supervised.py --data pubmed      $GENERIC_PARAM $k $LAYER_2 --weight_decay 1e-6 --model MLP --hidden 64 --dropout 0.3 
done


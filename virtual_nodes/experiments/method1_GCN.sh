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
  # GCN
  # python -u full-supervised.py --data cora        $GENERIC_PARAM $k $d $LAYER_2 --weight_decay 5e-4 --model GCN --hidden 16 
  # python -u full-supervised.py --data citeseer    $GENERIC_PARAM $k $d $LAYER_2 --weight_decay 5e-4 --model GCN --hidden 16 
  # python -u full-supervised.py --data pubmed      $GENERIC_PARAM $k $d $LAYER_2 --weight_decay 5e-5 --model GCN --hidden 16 
  python -u full-supervised.py --data chameleon   $GENERIC_PARAM $k $d $LAYER_2 --weight_decay 5e-5 --model GCN --hidden 16
  # python -u full-supervised.py --data cornell     $GENERIC_PARAM $k $d $LAYER_2 --weight_decay 5e-5 --model GCN --hidden 16 
  # python -u full-supervised.py --data texas       $GENERIC_PARAM $k $d $LAYER_2 --weight_decay 5e-5 --model GCN --hidden 16 
  # python -u full-supervised.py --data wisconsin   $GENERIC_PARAM $k $d $LAYER_2 --weight_decay 5e-5 --model GCN --hidden 16 
  # python -u full-supervised.py --data squirrel    $GENERIC_PARAM $k $d $LAYER_2 --weight_decay 5e-5 --model GCN --hidden 16 
  # python -u full-supervised.py --data film        $GENERIC_PARAM $k $d $LAYER_2 --weight_decay 5e-5 --model GCN --hidden 16 
done


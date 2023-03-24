#!/bin/bash

set -e

# Variables
GENERIC_PARAM="--splits 3 --n_seeds 3 --epochs 1500 --patience 100 --verbosity 0 --augment --augment_ratio 1.0 --include_vnode_labels --clip --khops 2"
LAYER_2="--layer 2"

# Operations
cd Heterophily_and_oversmoothing

echo "EXPERIMENT SET: ${GENERIC_PARAM} ${LAYER_2}"
# GAT 
python -u full-supervised.py --data cora        $GENERIC_PARAM $k $LAYER_2 --weight_decay 5e-4 --model GAT --hidden 8 --dropout 0.6 --lr 0.005
python -u full-supervised.py --data citeseer    $GENERIC_PARAM $k $LAYER_2 --weight_decay 5e-4 --model GAT --hidden 8 --dropout 0.6 --lr 0.005
python -u full-supervised.py --data chameleon   $GENERIC_PARAM $k $LAYER_2 --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005
python -u full-supervised.py --data cornell     $GENERIC_PARAM $k $LAYER_2 --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005            
python -u full-supervised.py --data texas       $GENERIC_PARAM $k $LAYER_2 --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005             
python -u full-supervised.py --data wisconsin   $GENERIC_PARAM $k $LAYER_2 --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005             
python -u full-supervised.py --data film        $GENERIC_PARAM $k $LAYER_2 --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005 --use_sparse
python -u full-supervised.py --data squirrel    $GENERIC_PARAM $k $LAYER_2 --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005 --use_sparse
python -u full-supervised.py --data pubmed      $GENERIC_PARAM $k $LAYER_2 --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005 --use_sparse



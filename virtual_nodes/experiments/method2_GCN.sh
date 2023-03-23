#!/bin/bash

set -e

# Variables
declare -a directed=("--directed" " ")
declare -a embedding_layer=("--use_embed" " ") # TODO remove
GENERIC_PARAM="--splits 3 --n_seeds 3 --epochs 1500 --patience 100 --verbosity 0 --clip --augment --augment_ratio 0.15"
GENERIC_PARAM="${GENERIC_PARAM} --learn_feats" # exclude vnode labels
LAYER_2="--layer 2"

# Operations
cd Heterophily_and_oversmoothing


python -u full-supervised.py --data cora        $GENERIC_PARAM $emb $d $LAYER_2 --weight_decay 5e-4 --model GCN --hidden 16 
python -u full-supervised.py --data citeseer    $GENERIC_PARAM $emb $d $LAYER_2 --weight_decay 5e-4 --model GCN --hidden 16 
python -u full-supervised.py --data pubmed      $GENERIC_PARAM $emb $d $LAYER_2 --weight_decay 5e-5 --model GCN --hidden 16 
python -u full-supervised.py --data chameleon   $GENERIC_PARAM $emb $d $LAYER_2 --weight_decay 5e-5 --model GCN --hidden 16 
python -u full-supervised.py --data cornell     $GENERIC_PARAM $emb $d $LAYER_2 --weight_decay 5e-5 --model GCN --hidden 16 
python -u full-supervised.py --data texas       $GENERIC_PARAM $emb $d $LAYER_2 --weight_decay 5e-5 --model GCN --hidden 16 
python -u full-supervised.py --data wisconsin   $GENERIC_PARAM $emb $d $LAYER_2 --weight_decay 5e-5 --model GCN --hidden 16 
python -u full-supervised.py --data squirrel    $GENERIC_PARAM $emb $d $LAYER_2 --weight_decay 5e-5 --model GCN --hidden 16 
python -u full-supervised.py --data film        $GENERIC_PARAM $emb $d $LAYER_2 --weight_decay 5e-5 --model GCN --hidden 16 
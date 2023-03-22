set -e

# Variables
declare -a directed=("--directed" " ")
declare -a embedding_layer=("--use_embed" " ")
GENERIC_PARAM="--splits 1 --n_seeds 1 --epochs 1500 --patience 100 --verbosity 0 --clip --augment --augment_ratio 0.15 --include_vnode_labels"
LAYER_2="--layer 2"

# Operations
cd Heterophily_and_oversmoothing

for d in "${directed[@]}"
do
  for emb in "${embedding_layer[@]}"
  do
    echo "EXPERIMENT SET: $d $emb"
    # GCN
    python -u full-supervised.py --data cora        ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 5e-4 --model GCN --hidden 16 
    python -u full-supervised.py --data citeseer    ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 5e-4 --model GCN --hidden 16 
    python -u full-supervised.py --data pubmed      ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 5e-5 --model GCN --hidden 16 
    python -u full-supervised.py --data chameleon   ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 5e-5 --model GCN --hidden 16 
    python -u full-supervised.py --data cornell     ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 5e-5 --model GCN --hidden 16 
    python -u full-supervised.py --data texas       ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 5e-5 --model GCN --hidden 16 
    python -u full-supervised.py --data wisconsin   ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 5e-5 --model GCN --hidden 16 
    python -u full-supervised.py --data squirrel    ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 5e-5 --model GCN --hidden 16 
    python -u full-supervised.py --data film        ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 5e-5 --model GCN --hidden 16 
    # MLP 
    python -u full-supervised.py --data cornell     ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 1e-4 --model MLP --hidden 16               
    python -u full-supervised.py --data texas       ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 1e-4 --model MLP --hidden 16               
    python -u full-supervised.py --data wisconsin   ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 1e-4 --model MLP --hidden 16               
    python -u full-supervised.py --data chameleon   ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 1e-8 --model MLP --hidden 16               
    python -u full-supervised.py --data cora        ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 1e-4 --model MLP --hidden 16               
    python -u full-supervised.py --data citeseer    ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 1e-7 --model MLP --hidden 64               
    python -u full-supervised.py --data film        ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 1e-3 --model MLP --hidden 16 --dropout 0   
    python -u full-supervised.py --data squirrel    ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 1e-4 --model MLP --hidden 16 --dropout 0.6 
    python -u full-supervised.py --data pubmed      ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 1e-6 --model MLP --hidden 64 --dropout 0.3 
    # GAT 
    python -u full-supervised.py --data cora        ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 5e-4 --model GAT --hidden 8 --dropout 0.6 --lr 0.005             
    python -u full-supervised.py --data citeseer    ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 5e-4 --model GAT --hidden 8 --dropout 0.6 --lr 0.005             
    python -u full-supervised.py --data chameleon   ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005             
    python -u full-supervised.py --data cornell     ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005             
    python -u full-supervised.py --data texas       ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005             
    python -u full-supervised.py --data wisconsin   ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005             
    python -u full-supervised.py --data film        ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005 --use_sparse
    python -u full-supervised.py --data squirrel    ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005 --use_sparse
    python -u full-supervised.py --data pubmed      ${GENERIC_PARAM} ${emb} ${d} ${LAYER_2} --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005 --use_sparse
    # GPRGNN 
    # python -u full-supervised.py --data cora 				${GENERIC_PARAM} ${emb} ${d} --layer 4 --lr 0.002 --dprate_GPRGNN 0.5 --weight_decay 5e-4 --alpha_GPRGNN 0.1 --model GPRGNN --hidden 64
    # python -u full-supervised.py --data citeseer 		${GENERIC_PARAM} ${emb} ${d} --layer 8 --lr 0.002 --dprate_GPRGNN 0.5 --weight_decay 5e-4 --alpha_GPRGNN 0.1 --model GPRGNN --hidden 64
    # python -u full-supervised.py --data pubmed 			${GENERIC_PARAM} ${emb} ${d} --layer 2 --lr 0.05  --dprate_GPRGNN 0.5 --weight_decay 5e-4 --alpha_GPRGNN 0.2 --model GPRGNN --hidden 64
    # python -u full-supervised.py --data chameleon 	${GENERIC_PARAM} ${emb} ${d} --layer 2 --lr 0.05  --dprate_GPRGNN 0.7 --weight_decay 0    --alpha_GPRGNN 1   --model GPRGNN --hidden 64
    # python -u full-supervised.py --data cornell 		${GENERIC_PARAM} ${emb} ${d} --layer 2 --lr 0.05  --dprate_GPRGNN 0.5 --weight_decay 5e-4 --alpha_GPRGNN 0.9 --model GPRGNN --hidden 64
    # python -u full-supervised.py --data texas 			${GENERIC_PARAM} ${emb} ${d} --layer 2 --lr 0.05  --dprate_GPRGNN 0.5 --weight_decay 5e-4 --alpha_GPRGNN 1   --model GPRGNN --hidden 64
    # python -u full-supervised.py --data wisconsin 	${GENERIC_PARAM} ${emb} ${d} --layer 2 --lr 0.05  --dprate_GPRGNN 0.5 --weight_decay 5e-4 --alpha_GPRGNN 1   --model GPRGNN --hidden 64
    # python -u full-supervised.py --data squirrel 		${GENERIC_PARAM} ${emb} ${d} --layer 2 --lr 0.05  --dprate_GPRGNN 0.7 --weight_decay 0    --alpha_GPRGNN 0   --model GPRGNN --hidden 64
    # python -u full-supervised.py --data film 				${GENERIC_PARAM} ${emb} ${d} --layer 2 --lr 0.01  --dprate_GPRGNN 0.5 --weight_decay 0    --alpha_GPRGNN 0.9 --model GPRGNN --hidden 64
  done
done #  > results.txt


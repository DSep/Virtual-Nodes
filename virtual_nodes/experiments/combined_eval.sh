# GCN
python -u full-supervised.py --data cora       --layer 2 --weight_decay 5e-4 --model GCN --hidden 16  --augment
python -u full-supervised.py --data citeseer   --layer 2 --weight_decay 5e-4 --model GCN --hidden 16  --augment
python -u full-supervised.py --data pubmed     --layer 2 --weight_decay 5e-5 --model GCN --hidden 16  --augment
python -u full-supervised.py --data chameleon  --layer 2 --weight_decay 5e-5 --model GCN --hidden 16  --augment
python -u full-supervised.py --data cornell    --layer 2 --weight_decay 5e-5 --model GCN --hidden 16  --augment
python -u full-supervised.py --data texas      --layer 2 --weight_decay 5e-5 --model GCN --hidden 16  --augment
python -u full-supervised.py --data wisconsin  --layer 2 --weight_decay 5e-5 --model GCN --hidden 16  --augment
python -u full-supervised.py --data squirrel   --layer 2 --weight_decay 5e-5 --model GCN --hidden 16  --augment
python -u full-supervised.py --data film       --layer 2 --weight_decay 5e-5 --model GCN --hidden 16  --augment

# MLP
python -u full-supervised.py --data cornell     --layer 2 --weight_decay 1e-4 --model MLP --hidden 16                --augment
python -u full-supervised.py --data texas       --layer 2 --weight_decay 1e-4 --model MLP --hidden 16                --augment
python -u full-supervised.py --data wisconsin   --layer 2 --weight_decay 1e-4 --model MLP --hidden 16                --augment
python -u full-supervised.py --data chameleon   --layer 2 --weight_decay 1e-8 --model MLP --hidden 16                --augment
python -u full-supervised.py --data cora        --layer 2 --weight_decay 1e-4 --model MLP --hidden 16                --augment
python -u full-supervised.py --data citeseer    --layer 2 --weight_decay 1e-7 --model MLP --hidden 64                --augment
python -u full-supervised.py --data film        --layer 2 --weight_decay 1e-3 --model MLP --hidden 16 --dropout 0    --augment
python -u full-supervised.py --data squirrel    --layer 2 --weight_decay 1e-4 --model MLP --hidden 16 --dropout 0.6  --augment
python -u full-supervised.py --data pubmed      --layer 2 --weight_decay 1e-6 --model MLP --hidden 64 --dropout 0.3  --augment

# GAT
python -u full-supervised.py --data cora        --layer 2 --weight_decay 5e-4 --model GAT --hidden 8 --dropout 0.6 --lr 0.005               --augment
python -u full-supervised.py --data citeseer    --layer 2 --weight_decay 5e-4 --model GAT --hidden 8 --dropout 0.6 --lr 0.005               --augment
python -u full-supervised.py --data chameleon   --layer 2 --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005               --augment
python -u full-supervised.py --data cornell     --layer 2 --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005               --augment
python -u full-supervised.py --data texas       --layer 2 --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005               --augment
python -u full-supervised.py --data wisconsin   --layer 2 --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005               --augment
python -u full-supervised.py --data film        --layer 2 --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005 --use_sparse  --augment
python -u full-supervised.py --data squirrel    --layer 2 --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005 --use_sparse  --augment
python -u full-supervised.py --data pubmed      --layer 2 --weight_decay 5e-5 --model GAT --hidden 8 --dropout 0.6 --lr 0.005 --use_sparse  --augment

# Can Virtual Nodes Improve Node Classification on Heterophilic Graphs? 

This repository contains the code for the untitled research project on virtual nodes in GNNs, by [Sepand Dyanatkar](https://github.com/DSep) and [Jamie Weigold](https://github.com/jweig0ld).

Setup and usage
```
git clone git@github.com:DSep/virtual-nodes.git
cd virtual-nodes
git submodule update --init --recursive
pip install -r requirements.txt
pip install -e Heterophily_and_oversmoothing
pip install -e understanding_oversquashing
pip install -e .
```

To avoid dependency conflicts, it is highly recommended that our code is run in a fresh conda/miniconda environment which can be easily configured (if running mac OSX) by executing the following in a terminal:

```
conda create -n virtual_nodes python=3.9
conda activate virtual_nodes
conda install pytorch torchvision -c pytorch
conda install pyg -c pyg
conda install -c dglteam dgl
```

## Running experiments
Start in the root directory of the repository
```bash
sh virtual_nodes/experiments/method1.sh
```

Or  on the server:
```bash
nohup bash virtual_nodes/experiments/method1.sh > TIME_DAY_03_2023_method1.log 2> nohup.err < /dev/null &
```

TODOS:
- [ ] Add setup.py files for top level and submodules to make easy referencing from anywhere (aka Heterophily and curvature folders)
- [ ] Setup proper testing in home directory and move tests to this folder

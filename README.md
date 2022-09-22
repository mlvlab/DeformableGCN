# Deformable Graph Convolutional Networks
This repository is the implementation of Deformable Graph Convolutional Networks~(Deformable GCNs).

> Jinyoung Park, Sungdong Yoo, Jihwan Park, Hyunwoo J. Kim, Deformable Graph Convolutional Networks, In AAAI Conference on Artificial Intelligence (AAAI) 2022.
> 


### Environmental Setup

```bash
# Python version : 3.8.13, Cuda version : 10.2
$ conda env create --file env.yaml
$ conda activate deformablegcn
```

### Running the code

| Arg | Description |
| --- | --- |
| —dataset | Dataset |
| —lr | Learning rate |
| —weight_decay | weight decay |
| —epochs | Number of epochs to train |
| —hidden | Dimensionality of hidden embeddings |
| —dropout | Dropout probability |
| —num_blocks | Number of blocks |
| —n_neighbor | Number of neighbors of latent neighborhood graphs |
| —n_hops | Number of hops (l) |
| —n_kernels | Number of kernels (k) |
| —alpha | Hyperparameter for separating regularization loss  |
| —beta | Hyperparameter for focusing regularization loss |
| —phi_dim | Dimensionality of phi |
| —split_idx | Index of splits provided by (Pei et al., 2020) |

For example, if you want to run on Cora dataset with 0-th split,

```
python main.py --dataset cora --split_idx 0
```

### Citation

if this work is useful for your research, please cite our paper:

```
@inproceedings{park2022deformable,
  title={Deformable Graph Convolutional Networks},
  author={Park, Jinyoung and Yoo, Sungdong and Park, Jihwan and Kim, Hyunwoo J},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={7},
  pages={7949--7956},
  year={2022}
}
```

### Acknowledgement

This repo is built upon the following work:

```
Geom-GCN: Geometric Graph Convolutional Networks. Hongbin Pei, Bingzhe Wei, Kevin Chen-Chuan Chang, Yu Lei, and Bo Yang. ICLR 2020.
Code : https://github.com/graphdml-uiuc-jlu/geom-gcn
```

# LACL

Original codebase for [Deep Contrastive Learning of Molecular Conformation for Efficient Property Prediction]()

Yang Jeong Park, HyunGi Kim, Jeonghee Jo and Sungroh Yoon   
Massachusetts Institute of Technology   
Seoul National University


![](images/Advantage_of_our_approach_revision.png)

>In this work, we propose a novel method called Local Atomic environment Contrastive Learning (LACL) which is a deep contrastive learning-based domain adaptation method. LACL effectively treats the trade-off between cost-effective conformation generation and prediction accuracy by minimizing the distance between molecular geometry embeddings instead of generating conformations directly. We demonstrate that our approach achieves quantum chemical accuracy without density functional theory (DFT) geometric relaxation, while also speeding up the inference time 100-fold faster than DFT optimization-based models.

# Instrallation
```
conda env create -f lacl.yaml
conda activate lacl
```
# Dataset
You can download datasets used in the paper [here]() and extract the zip file under `./data` folder. Both QM9 and QMugs should be saved in the folder under their name. Conformations of all the data is pickled after preprocessing. 
## [QM9](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904)
`qm9_all.pickle`   
List of dictionaries with properties. One dictionary corresponds to one molecule. It also contains cartesian coordinates of MMFF conformations and MMFF potential.
`qm9_all_cgcf.pkl`   
List of rdkit molecules with cartesian coordinates of CGCF-ConfGen conformations. They were calculated by official [CGCF-ConfGen](https://github.com/MinkaiXu/CGCF-ConfGen) implements.

# Training
```
python main.py
```
## Arguments explanations    
`--lacl`    
True for training LACL, False for training modified-ALIGNN    
`--loss`    
contrastive+prediction loss is default   
`--set`    
'src' for source domain and 'tgt' for target domain. LACL doesn't affect this. It's for modified-ALIGNN training.  
`--target`    
QM9: **mu**, alpha, **homo**, **lumo**, **gap**, r2, zpve, **U0**, U, **G**, H, and Cv.   
QMugs: **GFN2:DIPOLE**, **GFN2:HOMO_LUMO_GAP**, **GFN2:TOTAL_FREE_ENERGY**, [Target labels](data/data.py)   
`--geometry`   
Select target domain


# Acknowledgment
- Pytorch implementation of ALIGNN: https://github.com/usnistgov/alignn
- Self-supervised learning strategies for GNN: https://github.com/nerdslab/bgrl
- Data generation using CGCF-ConfGen: https://github.com/MinkaiXu/CGCF-ConfGen

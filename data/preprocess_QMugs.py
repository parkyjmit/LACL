import glob 
from pathlib import Path
from typing import Union
from rdkit import Chem
from rdkit.Chem import PandasTools
from tqdm import tqdm
from rdkit.Chem import AllChem
import pickle
from joblib import Parallel, delayed
import numpy as np
np.random.seed(123)

num = 40
def treat(
        base_dir: Union[str, Path] = Path(
            './QMugs/structures/'),
        target_dir: Union[str, Path] = Path(
            f'./QMugs/structures_{num}_energy/')
):
    # Target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # 
    for chembl_id_dir in tqdm(base_dir.iterdir()):
        if not chembl_id_dir.is_dir():
            continue
        chembl_id = chembl_id_dir.name

        energy = []
        conformations = []
        conformations_noH = []
        for conf in chembl_id_dir.iterdir():
            frame_noH = PandasTools.LoadSDF(
                str(conf),
                smilesName='SMILES',
                molColName='Molecule',
                includeFingerprints=False,
                removeHs=True
            )
            
            num_heavy_atoms = frame_noH.Molecule[0].GetNumAtoms()
            if num_heavy_atoms == num:  # number of heavy atoms
                frame = PandasTools.LoadSDF(
                    str(conf),
                    smilesName='SMILES',
                    molColName='Molecule',
                    includeFingerprints=False,
                    removeHs=False
                ) # save
                conformations.append(frame)
                conformations_noH.append(frame_noH)
                energy.append(frame_noH['GFN2:TOTAL_FREE_ENERGY'][0])  # save energy
        if energy != []: # compare energy and select conformation with minimum energy
            frame = conformations[energy.index(min(energy))]
            frame_noH = conformations_noH[energy.index(min(energy))]
            xyz = Chem.rdmolfiles.MolToXYZBlock(frame_noH.Molecule[0])  # conformation information

            conf_id = chembl_id + '_' + conf.name.replace('.sdf', '')
            file_name = conf_id + ".xyz"
            f = open(target_dir / file_name, "w+")
            f.write(xyz)
            f.close()
            file_name = conf_id + "_label.pkl"
            with open(target_dir / file_name, 'wb') as f:
                pickle.dump(frame[frame.columns], f)  # label information
            file_name = conf_id + "_label_noH.pkl"
            with open(target_dir / file_name, 'wb') as f:
                pickle.dump(frame_noH[frame_noH.columns], f)  # label information

if __name__ == '__main__':
    treat()
    molpkl_list = []
    for molpkl in tqdm(glob.glob(f'./QMugs/structures_{num}_energy/*_label.pkl')):
        with open(molpkl, 'rb') as f:
            mol = pickle.load(f)
        molpkl_list.append(mol)
    with open(f'./QMugs/QMugs_{num}_energy.pkl', 'wb') as f:
        pickle.dump(molpkl_list, f)

    def mol_convert_mmff(split):
        split_list = []
        for molpkl in tqdm(split):
            mol = molpkl.Molecule[0]   
            AllChem.MMFFOptimizeMolecule(mol, confId=0)
            split_list.append(mol)
        return split_list

    test_indices = np.random.choice(len(molpkl_list), int(len(molpkl_list) * 0.1), replace=False)
    test_molpkl_list = [molpkl_list[i] for i in test_indices]

    molpkl_split = []
    for i in range(18):
        start = int(len(molpkl_list)/180)*i
        if i == 17:
            end = len(test_molpkl_list)
        else:
            end = int(len(molpkl_list)/180)*(i+1)
        molpkl_split.append(test_molpkl_list[start:end])

    mol_mmff_list = []
    mol_mmff_split_list = Parallel(n_jobs=18)(delayed(mol_convert_mmff)(split) for split in molpkl_split)
    for split in mol_mmff_split_list:
        mol_mmff_list += split
        
    with open(f'./data/QMugs/QMugs_{num}_energy_test.pkl', 'wb') as f:
        pickle.dump(test_molpkl_list, f)
    with open(f'./data/QMugs/QMugs_{num}_energy_mmff.pkl', 'wb') as f:
        pickle.dump(mol_mmff_list, f)
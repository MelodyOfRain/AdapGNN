import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
import math
import os
import numpy as np

PARA_DIR = os.path.join("../data")

class Descriptor(object):
    def __init__(self) -> None:
        self.locater = group_locater()
        # this idx is obetained by radnomly select the structure in logP.txt
        self.masked_idx = [
            69,
            9,
            48,
            64,
            66,
            81,
            47,
            59,
            80,
            72,
            38,
            34,
            4,
            90,
            93,
            18,
            31,
            11,
            25,
            39,
            14,
            55,
            7,
            49,
            77,
            15,
            60,
            67,
            44,
            10,
            8,
            92,
            84,
            0,
            56,
            3,
            75,
            95,
            43,
            28,
            54,
            33,
            82,
            19,
            70,
            5,
            24,
            94,
            74,
            20,
        ]

        self.collection = CrippenParamCollection(self.masked_idx).d_params

        self.table = Chem.GetPeriodicTable()

        self.electronegativity = dict(
            C=2.55,
            N=3.04,
            O=3.44,
            F=3.98,
            Cl=3.16,
            Br=2.96,
            I=2.66,
            S=2.58,
            P=2.19,
            As=2.18,
            Se=2.48,
            Si=1.90,
            B=2.04,
        )

        self.symbols_list = [
            "B",
            "C",
            "N",
            "O",
            "F",
            "Si",
            "P",
            "S",
            "Cl",
            "As",
            "Se",
            "Br",
            "Te",
            "I",
            "At",
            "other",
        ]

        # log(e) compute by slatter rules
        self.atomic_energy = {
            "B": -5.778914684004642,
            "C": -6.169143049923254,
            "N": -6.495594806863924,
            "O": -6.7762019287412425,
            "F": -7.022261841747614,
            "Si": -7.975387816153126,
            "P": -8.123823586879622,
            "S": -8.262193652878077,
            "Cl": -8.391754437892535,
            "As": -9.898375953000498,
            "Se": -9.961788606021466,
            "Br": -10.02332203263317,
            "Te": -10.894049839926026,
            "I": -10.93485466953677,
            "At": -12.190744437816809,
        }

    def compute_atom_num(self, smiles: str, symbols: list, reverse: bool = False):
        """compute the number of atoms with given symbols
        Args:
            smiles (str): smiles of molecule
            symbols (list): atoms should be foucus on
            reverse (bool, optional): focus on the rest catrgeries of atoms with given symbols. Defaults to False.

        Returns:
            node_mask: binary mask of atoms
            y: computed descriptor
        """
        mol = Chem.MolFromSmiles(smiles)
        atoms = [a for a in mol.GetAtoms()]
        node_mask = torch.zeros(len(atoms))
        contribs = torch.zeros(len(atoms))
        for i, atom in enumerate(atoms):
            if reverse:
                if atom.GetSymbol() not in symbols:
                    node_mask[i] = 1
                    contribs[i] = 1
            else:
                if atom.GetSymbol() in symbols:
                    node_mask[i] = 1
                    contribs[i] = 1
        node_mask = node_mask.unsqueeze(0)
        y = torch.sum(contribs)
        return node_mask.long(), contribs, y

    def compute_protons_num(self, smiles: str, symbols: list, reverse: bool = False):
        """ compute the number of protons in a molecule

        Args:
            smiles (str): smiles of molecule
            symbols (list): atoms should be foucus on
            reverse (bool, optional): focus on the rest catrgeries of atoms with given symbols. Defaults to False.

        Returns:
            node_mask: binary mask of atoms
            y: computed descriptor
        """        

        mol = Chem.MolFromSmiles(smiles)
        atoms = [a for a in mol.GetAtoms()]
        node_mask = torch.zeros(len(atoms))
        contribs = torch.zeros(len(atoms))
        for i, atom in enumerate(atoms):
            if reverse:
                if atom.GetSymbol() not in symbols:
                    node_mask[i] = 1
                    contribs[i] = atom.GetAtomicNum()
            else:
                if atom.GetSymbol() in symbols:
                    node_mask[i] = 1
                    contribs[i] = atom.GetAtomicNum()
        y = torch.sum(contribs)
        return node_mask.long(), contribs, y

    def compute_molecule_energy(self, smiles: str, symbols: list):
        """_summary_

        Args:
            smiles (str): _description_
            symbols (list): _description_

        Returns:
            _type_: _description_
        """        

        mol = Chem.MolFromSmiles(smiles)
        atoms = [a for a in mol.GetAtoms()]
        node_mask = torch.zeros(len(atoms))
        contribs = torch.zeros(len(atoms))
        for i, atom in enumerate(atoms):
            if (
                atom.GetSymbol() in symbols
                and atom.GetSymbol() in self.atomic_energy.keys()
            ):
                node_mask[i] = 1
                contribs[i] = self.atomic_energy[atom.GetSymbol()]
        y = torch.sum(contribs)
        return node_mask.long(), contribs, y

    def compute_bond_num(self, smiles: str, bondtypes: list, reverse: bool = False):
        """_summary_

        Args:
            smiles (str): _description_
            bondtypes (list): _description_
            reverse (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """        
        mol = Chem.MolFromSmiles(smiles)
        bonds = [b for b in mol.GetBonds()]
        node_mask = torch.zeros(mol.GetNumAtoms())
        contribs = torch.zeros(mol.GetNumAtoms())
        for i, bond in enumerate(bonds):
            idxs = [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]
            bt = int(bond.GetBondType())
            if reverse:
                if bt not in bondtypes:
                    node_mask[idxs] = 1
                    contribs[idxs] += 1 / 2
            else:
                if bt in bondtypes:
                    node_mask[idxs] = 1
                    contribs[idxs] += 1 / 2
        y = torch.sum(contribs)
        return node_mask.long(), contribs, y

    def compute_polar_coefficient(self, smiles: str):
        """_summary_

        Args:
            smiles (str): _description_

        Returns:
            _type_: _description_
        """        
        mol = Chem.MolFromSmiles(smiles)
        node_mask = torch.zeros(mol.GetNumAtoms())
        contribs = torch.zeros(mol.GetNumAtoms())
        y = 0
        for i, bond in enumerate(mol.GetBonds()):
            bondtype = int(bond.GetBondType())
            symbols = [bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()]
            if len(set(symbols)) != 1:
                idxs = [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]
                if (
                    symbols[0] in self.electronegativity.keys()
                    and symbols[1] in self.electronegativity.keys()
                    and bondtype in [1, 2, 3, 12]
                ):
                    charge = [self.electronegativity[s] for s in symbols]
                    charge = abs(charge[0] - charge[1])
                    if bondtype == 12:
                        coef = 3
                    else:
                        coef = bondtype * 2
                    contribs[idxs] += charge * coef / 2
                    node_mask[idxs] = 1
        y = torch.sum(contribs)
        return node_mask, contribs, y

    def compute_aromatic_atom_num(self, smiles: str):
        """_summary_

        Args:
            smiles (str): _description_

        Returns:
            _type_: _description_
        """        
        mol = Chem.MolFromSmiles(smiles)
        atoms = [a for a in mol.GetAtoms()]
        node_mask = torch.zeros(len(atoms))
        contribs = torch.zeros(len(atoms))
        for i, atom in enumerate(atoms):
            if atom.GetIsAromatic():
                node_mask[i] = 1
                contribs[i] = 1
        y = torch.sum(contribs)
        return node_mask, contribs, y

    def compute_inring_atom_num(self, smiles: str):
        """_summary_

        Args:
            smiles (str): _description_

        Returns:
            _type_: _description_
        """        
        mol = Chem.MolFromSmiles(smiles)
        atoms = [a for a in mol.GetAtoms()]
        node_mask = torch.zeros(len(atoms))
        contribs = torch.zeros(len(atoms))
        for i, atom in enumerate(atoms):
            if atom.IsInRing():
                node_mask[i] = 1
                contribs[i] = 1
        y = torch.sum(contribs)
        return node_mask, contribs, y

    def compute_group_num(self, smiles: str):
        """_summary_

        Args:
            smiles (str): _description_

        Returns:
            _type_: _description_
        """        
        node_mask, contribs, y = self.locater.all_loc(smiles)
        return node_mask, contribs, y

    def compute_tpsa(self, smiles: str, includeSandP: bool = True):
        """_summary_

        Args:
            smiles (str): _description_
            includeSandP (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """        
        mol = Chem.MolFromSmiles(smiles)
        nAtoms = mol.GetNumAtoms()
        node_mask = torch.zeros(nAtoms)
        contribs = torch.zeros(nAtoms)
        nNbrs = [0] * nAtoms
        nSing = [0] * nAtoms
        nDoub = [0] * nAtoms
        nTrip = [0] * nAtoms
        nArom = [0] * nAtoms
        nHs = [0] * nAtoms
        for bnd in mol.GetBonds():
            if bnd.GetBeginAtom().GetAtomicNum() == 1:
                nNbrs[bnd.GetEndAtomIdx()] -= 1
                nHs[bnd.GetEndAtomIdx()] += 1
            elif bnd.GetEndAtom().GetAtomicNum() == 1:
                nNbrs[bnd.GetBeginAtomIdx()] -= 1
                nHs[bnd.GetBeginAtomIdx()] += 1
            elif bnd.GetIsAromatic():
                nArom[bnd.GetBeginAtomIdx()] += 1
                nArom[bnd.GetEndAtomIdx()] += 1
            else:
                if bnd.GetBondType() == BondType.SINGLE:
                    nSing[bnd.GetBeginAtomIdx()] += 1
                    nSing[bnd.GetEndAtomIdx()] += 1
                elif bnd.GetBondType() == BondType.DOUBLE:
                    nDoub[bnd.GetBeginAtomIdx()] += 1
                    nDoub[bnd.GetEndAtomIdx()] += 1
                elif bnd.GetBondType() == BondType.TRIPLE:
                    nTrip[bnd.GetBeginAtomIdx()] += 1
                    nTrip[bnd.GetEndAtomIdx()] += 1

        for i in range(nAtoms):
            atom = mol.GetAtomWithIdx(i)
            atNum = atom.GetAtomicNum()

            if (
                atNum != 7
                and atNum != 8
                and (not includeSandP or (atNum != 15 and atNum != 16))
            ):
                continue

            nHs[i] += atom.GetTotalNumHs()
            chg = atom.GetFormalCharge()
            in3Ring = mol.GetRingInfo().IsAtomInRingOfSize(i, 3)
            nNbrs[i] += atom.GetDegree()

            contribs[i] = -1
            if atNum == 7:
                if nNbrs[i] == 1:
                    if nHs[i] == 0 and chg == 0 and nTrip[i] == 1:
                        contribs[i] = 23.79
                        node_mask[i] = 1
                    elif nHs[i] == 1 and chg == 0 and nDoub[i] == 1:
                        contribs[i] = 23.85
                        node_mask[i] = 1
                    elif nHs[i] == 2 and chg == 0 and nSing[i] == 1:
                        contribs[i] = 26.02
                        node_mask[i] = 1
                    elif nHs[i] == 2 and chg == 1 and nDoub[i] == 1:
                        contribs[i] = 25.59
                        node_mask[i] = 1
                    elif nHs[i] == 3 and chg == 1 and nSing[i] == 1:
                        contribs[i] = 27.64
                        node_mask[i] = 1
                elif nNbrs[i] == 2:
                    if nHs[i] == 0 and chg == 0 and nSing[i] == 1 and nDoub[i] == 1:
                        contribs[i] = 12.36
                        node_mask[i] = 1
                    elif nHs[i] == 0 and chg == 0 and nTrip[i] == 1 and nDoub[i] == 1:
                        contribs[i] = 13.60
                        node_mask[i] = 1
                    elif nHs[i] == 1 and chg == 0 and nSing[i] == 2 and in3Ring:
                        contribs[i] = 21.94
                        node_mask[i] = 1
                    elif nHs[i] == 1 and chg == 0 and nSing[i] == 2 and not in3Ring:
                        contribs[i] = 12.03
                        node_mask[i] = 1
                    elif nHs[i] == 0 and chg == 1 and nTrip[i] == 1 and nSing[i] == 1:
                        contribs[i] = 4.36
                        node_mask[i] = 1
                    elif nHs[i] == 1 and chg == 1 and nDoub[i] == 1 and nSing[i] == 1:
                        contribs[i] = 13.97
                        node_mask[i] = 1
                    elif nHs[i] == 2 and chg == 1 and nSing[i] == 2:
                        contribs[i] = 16.61
                        node_mask[i] = 1
                    elif nHs[i] == 0 and chg == 0 and nArom[i] == 2:
                        contribs[i] = 12.89
                        node_mask[i] = 1
                    elif nHs[i] == 1 and chg == 0 and nArom[i] == 2:
                        contribs[i] = 15.79
                        node_mask[i] = 1
                    elif nHs[i] == 1 and chg == 1 and nArom[i] == 2:
                        contribs[i] = 14.14
                        node_mask[i] = 1
                elif nNbrs[i] == 3:
                    if nHs[i] == 0 and chg == 0 and nSing[i] == 3 and in3Ring:
                        contribs[i] = 3.01
                        node_mask[i] = 1
                    elif nHs[i] == 0 and chg == 0 and nSing[i] == 3 and not in3Ring:
                        contribs[i] = 3.24
                        node_mask[i] = 1
                    elif nHs[i] == 0 and chg == 0 and nSing[i] == 1 and nDoub[i] == 2:
                        contribs[i] = 11.68
                        node_mask[i] = 1
                    elif nHs[i] == 0 and chg == 1 and nSing[i] == 2 and nDoub[i] == 1:
                        contribs[i] = 3.01
                        node_mask[i] = 1
                    elif nHs[i] == 1 and chg == 1 and nSing[i] == 3:
                        contribs[i] = 4.44
                        node_mask[i] = 1
                    elif nHs[i] == 0 and chg == 0 and nArom[i] == 3:
                        contribs[i] = 4.41
                        node_mask[i] = 1
                    elif nHs[i] == 0 and chg == 0 and nSing[i] == 1 and nArom[i] == 2:
                        contribs[i] = 4.93
                        node_mask[i] = 1
                    elif nHs[i] == 0 and chg == 0 and nDoub[i] == 1 and nArom[i] == 2:
                        contribs[i] = 8.39
                        node_mask[i] = 1
                    elif nHs[i] == 0 and chg == 1 and nArom[i] == 3:
                        contribs[i] = 4.10
                        node_mask[i] = 1
                    elif nHs[i] == 0 and chg == 1 and nSing[i] == 1 and nArom[i] == 2:
                        contribs[i] = 3.88
                        node_mask[i] = 1
                elif nNbrs[i] == 4:
                    if nHs[i] == 0 and nSing[i] == 4 and chg == 1:
                        contribs[i] = 0.0
                if contribs[i] < 0.0:
                    contribs[i] = 30.5 - nNbrs[i] * 8.2 + nHs[i] * 1.5
                    if contribs[i] < 0:
                        contribs[i] = 0.0
                    else:
                        node_mask[i] = 1
            elif atNum == 8:
                if nNbrs[i] == 1:
                    if nHs[i] == 0 and chg == 0 and nDoub[i] == 1:
                        contribs[i] = 17.07
                        node_mask[i] = 1
                    elif nHs[i] == 1 and chg == 0 and nSing[i] == 1:
                        contribs[i] = 20.23
                        node_mask[i] = 1
                    elif nHs[i] == 0 and chg == -1 and nSing[i] == 1:
                        contribs[i] = 23.06
                        node_mask[i] = 1
                elif nNbrs[i] == 2:
                    if nHs[i] == 0 and chg == 0 and nSing[i] == 2 and in3Ring:
                        contribs[i] = 12.53
                        node_mask[i] = 1
                    elif nHs[i] == 0 and chg == 0 and nSing[i] == 2 and not in3Ring:
                        contribs[i] = 9.23
                        node_mask[i] = 1
                    elif nHs[i] == 0 and chg == 0 and nArom[i] == 2:
                        contribs[i] = 13.14
                        node_mask[i] = 1
                if contribs[i] < 0.0:
                    contribs[i] = 28.5 - nNbrs[i] * 8.6 + nHs[i] * 1.5
                    if contribs[i] < 0:
                        contribs[i] = 0.0
                    else:
                        node_mask[i] = 1
            elif includeSandP and atNum == 15:
                contribs[i] = 0.0
                if nNbrs[i] == 2:
                    if nHs[i] == 0 and chg == 0 and nSing[i] == 1 and nDoub[i] == 1:
                        contribs[i] = 34.14
                        node_mask[i] = 1
                elif nNbrs[i] == 3:
                    if nHs[i] == 0 and chg == 0 and nSing[i] == 3:
                        contribs[i] = 13.59
                        node_mask[i] = 1
                    elif nHs[i] == 1 and chg == 0 and nSing[i] == 2 and nDoub[i] == 1:
                        contribs[i] = 23.47
                        node_mask[i] = 1
                elif nNbrs[i] == 4:
                    if nHs[i] == 0 and chg == 0 and nSing[i] == 3 and nDoub[i] == 1:
                        contribs[i] = 9.81
                        node_mask[i] = 1
                if contribs[i] < 0.0:
                    contribs[i] = 0.0
                else:
                    node_mask[i] = 1
            elif includeSandP and atNum == 16:
                contribs[i] = 0.0
                if nNbrs[i] == 1:
                    if nHs[i] == 0 and chg == 0 and nDoub[i] == 1:
                        contribs[i] = 32.09
                        node_mask[i] = 1
                    elif nHs[i] == 1 and chg == 0 and nSing[i] == 1:
                        contribs[i] = 38.80
                        node_mask[i] = 1
                elif nNbrs[i] == 2:
                    if nHs[i] == 0 and chg == 0 and nSing[i] == 2:
                        contribs[i] = 25.30
                        node_mask[i] = 1
                    elif nHs[i] == 0 and chg == 0 and nArom[i] == 2:
                        contribs[i] = 28.24
                        node_mask[i] = 1
                elif nNbrs[i] == 3:
                    if nHs[i] == 0 and chg == 0 and nArom[i] == 2 and nDoub[i] == 1:
                        contribs[i] = 21.70
                        node_mask[i] = 1
                    elif nHs[i] == 0 and chg == 0 and nSing[i] == 2 and nDoub[i] == 1:
                        contribs[i] = 19.21
                        node_mask[i] = 1
                elif nNbrs[i] == 4:
                    if nHs[i] == 0 and chg == 0 and nSing[i] == 2 and nDoub[i] == 2:
                        contribs[i] = 8.38
                        node_mask[i] = 1
                if contribs[i] < 0.0:
                    contribs[i] = 0.0
                else:
                    node_mask[i] = 1
        y = torch.sum(contribs)
        return node_mask, contribs, y

    def compute_asa(self, smiles: str, symbols: list, includeHs=False):
        """_summary_

        Args:
            smiles (str): _description_
            symbols (list): _description_
            includeHs (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """        
        mol = Chem.MolFromSmiles(smiles)
        nAtoms = mol.GetNumAtoms()
        rads = [0] * nAtoms
        node_mask = torch.zeros(nAtoms)
        contribs = torch.zeros(nAtoms)
        for i in range(nAtoms):
            rads[i] = self.table.GetRb0(mol.GetAtomWithIdx(i).GetAtomicNum())

        bondScaleFacts = [0.1, 0, 0.2, 0.3]
        for bondIt in mol.GetBonds():
            Ri = rads[bondIt.GetBeginAtomIdx()]
            Rj = rads[bondIt.GetEndAtomIdx()]
            bij = Ri + Rj
            if not bondIt.GetIsAromatic():
                if bondIt.GetBondType() < 4:
                    bij -= bondScaleFacts[bondIt.GetBondType()]
            else:
                bij -= bondScaleFacts[0]
            dij = min(max(abs(Ri - Rj), bij), Ri + Rj)
            contribs[bondIt.GetBeginAtomIdx()] += (
                Rj * Rj - (Ri - dij) * (Ri - dij) / dij
            )
            contribs[bondIt.GetEndAtomIdx()] += Ri * Ri - (Rj - dij) * (Rj - dij) / dij

        hContrib = 0.0
        if includeHs:
            Rj = self.table.GetRb0(1)
            for i in range(nAtoms):
                Ri = rads[i]
                bij = Ri + Rj
                dij = min(max(abs(Ri - Rj), bij), Ri + Rj)
                contribs[i] += Rj * Rj - (Ri - dij) * (Ri - dij) / dij
                hContrib += Ri * Ri - (Rj - dij) * (Rj - dij) / dij
        res = 0.0
        for i in range(nAtoms):
            if mol.GetAtomWithIdx(i).GetSymbol() not in symbols:
                Ri = rads[i]
                contribs[i] = math.pi * Ri * (4.0 * Ri - contribs[i])
                node_mask[i] = 1

        if includeHs and abs(hContrib) > 1e-4:
            Rj = self.table.GetRb0(1)
            hContrib = math.pi * Rj * (4.0 * Rj - hContrib)
        y = torch.sum(contribs * node_mask)
        return node_mask, contribs, y

    def compute_logp(self, smiles: str, includeHs=False):
        """_summary_

        Args:
            smiles (str): _description_
            includeHs (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """        
        workMol = Chem.MolFromSmiles(smiles)
        nAtoms = workMol.GetNumAtoms()
        node_mask = torch.ones(nAtoms)
        contribs = torch.zeros(nAtoms)
        if includeHs:
            workMol = Chem.AddHs(workMol, False, False)
        params = self.collection
        for param in params:
            if param.mask == 1:
                matches = workMol.GetSubstructMatches(param.dp_pattern, False, False)
                for match in matches:
                    idx = match[0]
                    if node_mask[idx]:
                        node_mask[idx] = 0
                        contribs[idx] = param.logp
                if sum(node_mask) == 0:
                    break
        if includeHs:
            workMol = None
        y = torch.sum(contribs)
        return 1 - node_mask, contribs, y

class group_locater(object):
    
    def __init__(self):
        super(group_locater, self).__init__()
        self.group_names = ["co", "cooh", "coo", "con", "n", "oh", "coc", "c3n"]

    def loc_co(self, mol):
        node_mask = torch.zeros(mol.GetNumAtoms())
        contribs = torch.zeros(mol.GetNumAtoms())
        for i, atom in enumerate(mol.GetAtoms()):
            bonds = [int(b.GetBondType()) for b in atom.GetBonds()]
            if atom.GetSymbol() == "O" and 2 in bonds and node_mask[i] == 0:
                catom = atom.GetNeighbors()[0]
                if catom.GetSymbol() == "C":
                    nes = catom.GetNeighbors()
                    nes = filter(lambda x: x.GetIdx() != atom.GetIdx(), nes)
                    nes = [n for n in nes]
                    syms = [a.GetSymbol() for a in nes]
                    if set(syms) == set(["C"]):
                        atom_idx = [catom.GetIdx(), atom.GetIdx()] + [
                            a.GetIdx() for a in nes
                        ]
                        contribs[atom_idx] += 1 / 4
                        node_mask[atom_idx] = 1
        y = torch.sum(contribs)
        return node_mask, contribs, y

    def loc_cooh(self, mol):
        node_mask = torch.zeros(mol.GetNumAtoms())
        contribs = torch.zeros(mol.GetNumAtoms())
        for i, atom in enumerate(mol.GetAtoms()):
            bonds = [int(b.GetBondType()) for b in atom.GetBonds()]
            if atom.GetSymbol() == "O" and 2 in bonds and node_mask[i] == 0:
                catom = atom.GetNeighbors()[0]
                if catom.GetSymbol() == "C":
                    nes = catom.GetNeighbors()
                    nes = filter(lambda x: x.GetIdx() != atom.GetIdx(), nes)
                    nes = [n for n in nes]
                    syms = [a.GetSymbol() for a in nes]
                    if "O" in syms and nes[syms.index("O")].GetTotalNumHs() == 1:
                        oatom = nes[syms.index("O")]
                        atom_idx = [catom.GetIdx()] + [oatom.GetIdx()] + [atom.GetIdx()]
                        contribs[atom_idx] += 1 / 3
                        node_mask[atom_idx] = 1
        y = torch.sum(contribs)
        return node_mask, contribs, y

    def loc_coo(self, mol):
        node_mask = torch.zeros(mol.GetNumAtoms())
        contribs = torch.zeros(mol.GetNumAtoms())
        for i, atom in enumerate(mol.GetAtoms()):
            bonds = [int(b.GetBondType()) for b in atom.GetBonds()]
            if atom.GetSymbol() == "O" and 2 in bonds and node_mask[i] == 0:
                catom = atom.GetNeighbors()[0]
                if catom.GetSymbol() == "C":
                    nes = catom.GetNeighbors()
                    nes = filter(lambda x: x.GetIdx() != atom.GetIdx(), nes)
                    nes = [n for n in nes]
                    syms = [a.GetSymbol() for a in nes]
                    if "O" in syms:
                        oatom = nes[syms.index("O")]
                        if [a.GetSymbol() for a in oatom.GetNeighbors()] == ["C", "C"]:
                            atom_idx = (
                                [oatom.GetIdx()]
                                + [atom.GetIdx()]
                                + [a.GetIdx() for a in oatom.GetNeighbors()]
                            )
                            node_mask[atom_idx] = 1
                            contribs[atom_idx] += 1 / 3
                            node_mask[atom_idx] = 1
        y = torch.sum(contribs)
        return node_mask, contribs, y

    def loc_con(self, mol):
        node_mask = torch.zeros(mol.GetNumAtoms())
        contribs = torch.zeros(mol.GetNumAtoms())
        for i, atom in enumerate(mol.GetAtoms()):
            bonds = [int(b.GetBondType()) for b in atom.GetBonds()]
            if atom.GetSymbol() == "O" and 2 in bonds and node_mask[i] == 0:
                catom = atom.GetNeighbors()[0]
                if catom.GetSymbol() == "C":
                    nes = catom.GetNeighbors()
                    nes = filter(lambda x: x.GetSymbol() == "N", nes)
                    nes = [n for n in nes]
                    atom_idx = [catom.GetIdx(), atom.GetIdx()]
                    for natom in nes:
                        atom_idx = [catom.GetIdx(), atom.GetIdx(), natom.GetIdx()]
                        contribs[atom_idx] += 1 / 3
                        node_mask[atom_idx] = 1
        y = torch.sum(contribs)
        return node_mask, contribs, y

    def loc_n(self, mol):
        node_mask = torch.zeros(mol.GetNumAtoms())
        contribs = torch.zeros(mol.GetNumAtoms())
        for i, atom in enumerate(mol.GetAtoms()):
            bonds = [int(b.GetBondType()) for b in atom.GetBonds()]
            if atom.GetSymbol() == "N" and set(bonds) == {1} and node_mask[i] == 0:
                nes = atom.GetNeighbors()
                nes = filter(lambda x: x.GetSymbol() == "C", nes)
                flag = 0
                for catom in nes:
                    bonds = [b for b in catom.GetBonds()]
                    bonds = filter(lambda x: int(x.GetBondType()) == 2, bonds)
                    bonds = [b for b in bonds]
                    if bonds != []:
                        bond_syms = [
                            bonds[0].GetBeginAtom().GetSymbol(),
                            bonds[0].GetEndAtom().GetSymbol(),
                        ]
                        if "O" in bond_syms:
                            flag = 1
                            break
                if flag == 1:
                    continue
                else:
                    # node_mask[atom.GetIdx()]=1
                    atom_idx = [atom.GetIdx()] + [a.GetIdx() for a in nes]
                    node_mask[atom_idx] = 1
                    contribs[atom_idx] += 1 / len(atom_idx)
                    node_mask[atom_idx] = 1
        y = torch.sum(contribs)
        return node_mask, contribs, y

    def loc_oh(self, mol):
        node_mask = torch.zeros(mol.GetNumAtoms())
        contribs = torch.zeros(mol.GetNumAtoms())
        for i, atom in enumerate(mol.GetAtoms()):
            if (
                atom.GetSymbol() == "O"
                and atom.GetTotalNumHs() == 1
                and node_mask[i] == 0
            ):
                tmp = atom.GetNeighbors()
                if len(tmp) != 0:
                    catom = tmp[0]
                    if catom.GetSymbol() == "C":
                        nes = catom.GetNeighbors()
                        nes = filter(lambda x: x.GetIdx() != atom.GetIdx(), nes)
                        nes = [n for n in nes]
                        syms = [a.GetSymbol() for a in nes]
                        if "O" not in syms:
                            atom_idx = [atom.GetIdx(), catom.GetIdx()]
                            node_mask[atom_idx] = 1
                            contribs[atom_idx] += 1 / 2
                            node_mask[atom_idx] = 1
        y = torch.sum(contribs)
        return node_mask, contribs, y

    def loc_coc(self, mol):
        node_mask = torch.zeros(mol.GetNumAtoms())
        contribs = torch.zeros(mol.GetNumAtoms())
        for i, atom in enumerate(mol.GetAtoms()):
            if (
                atom.GetSymbol() == "O"
                and len(atom.GetBonds()) == 2
                and node_mask[i] == 0
            ):
                nes = atom.GetNeighbors()
                syms = [a.GetSymbol() for a in nes]
                if set(syms) == set(["C"]):
                    flag = 0
                    for catom in nes:
                        bonds = [b for b in catom.GetBonds()]
                        bonds = filter(lambda x: int(x.GetBondType()) == 2, bonds)
                        bonds = [b for b in bonds]
                        if bonds != []:
                            bond_syms = [
                                bonds[0].GetBeginAtom().GetSymbol(),
                                bonds[0].GetEndAtom().GetSymbol(),
                            ]
                            if "O" in bond_syms:
                                flag = 1
                                break
                    if flag == 1:
                        continue
                    else:
                        atom_idx = [atom.GetIdx()] + [a.GetIdx() for a in nes]
                        node_mask[atom_idx] = 1
                        contribs[atom_idx] += 1 / 3
                        node_mask[atom_idx] = 1
        y = torch.sum(contribs)
        return node_mask, contribs, y

    def loc_c3n(self, mol):
        node_mask = torch.zeros(mol.GetNumAtoms())
        contribs = torch.zeros(mol.GetNumAtoms())
        for i, atom in enumerate(mol.GetAtoms()):
            bonds = [int(b.GetBondType()) for b in atom.GetBonds()]
            if atom.GetSymbol() == "N" and bonds == [3] and node_mask[i] == 0:
                catom = atom.GetNeighbors()[0]
                atom_idx = [atom.GetIdx(), catom.GetIdx()]
                node_mask[atom_idx] = 1
                contribs[atom_idx] += 1 / 2
                node_mask[atom_idx] = 1
        y = torch.sum(contribs)
        return node_mask, contribs, y

    def all_loc(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        methods = ["loc_" + method for method in self.group_names]
        node_masks = []
        contribs = []
        ys = []
        for method in methods:
            node_mask, contrib, y = getattr(self, method)(mol)
            node_masks.append(node_mask)
            contribs.append(contrib.view(-1, 1))
            ys.append(y)
        ys = torch.sum(torch.tensor(ys))
        node_masks = torch.stack(node_masks)
        contribs = torch.stack(contribs, 1)
        node_masks = torch.sum(node_masks, 0)
        node_masks[node_masks > 1] = 1
        contribs = torch.sum(contribs, 1)
        return node_masks, contribs, ys

class CrippenParamCollection:
    def __init__(self, masked_idx):
        self.d_params = []
        with open(os.path.join(PARA_DIR, "logP.txt"), "r") as f:
            inStream = f.read()
        inStream = inStream.splitlines()
        idx = 0
        for inLine in inStream:
            if inLine[0] != "#":
                paramObj = CrippenParams()
                paramObj.idx = idx
                tokens = inLine.split("\t")
                paramObj.label = tokens[0]
                paramObj.smarts = tokens[1]
                paramObj.dp_pattern = Chem.MolFromSmarts(paramObj.smarts)
                if idx not in masked_idx:
                    if tokens[2] != "":
                        paramObj.logp = float(tokens[2])
                    else:
                        paramObj.logp = 0.0
                    if tokens[3] != "":
                        try:
                            paramObj.mr = float(tokens[3])
                        except ValueError:
                            paramObj.mr = 0.0
                    else:
                        paramObj.mr = 0.0
                    paramObj.mask = 1
                self.d_params.append(paramObj)
                idx += 1

class CrippenParams:
    def __init__(self):
        self.idx = 0
        self.label = ""
        self.smarts = ""
        self.logp = 0.0
        self.mr = 0.0
        self.dp_pattern = None
        self.mask = 0

    def __del__(self):
        self.dp_pattern = None

electron_configuration = {
    "B": [2, 3],
    "C": [2, 4],
    "N": [2, 5],
    "O": [2, 6],
    "F": [2, 7],
    "Si": [2, 8, 4],
    "P": [2, 8, 5],
    "S": [2, 8, 6],
    "Cl": [2, 8, 7],
    "As": [2, 8, 8, 0, 10, 5],
    "Se": [2, 8, 8, 0, 10, 6],
    "Br": [2, 8, 8, 0, 10, 7],
    "Te": [2, 8, 8, 10, 8, 10, 0, 0, 6],
    "I": [2, 8, 8, 10, 8, 10, 0, 0, 7],
    "At": [2, 8, 8, 10, 8, 10, 14, 8, 10, 14, 7],
}

n_star = [1, 2, 3, 3.7, 4.0, 4.2]

def compute_atomic_energy(atom):
    atom_config = electron_configuration[atom.GetSymbol()]
    e = 0
    sigma = [0] * len(atom_config)
    for i in range(len(atom_config)):
        if atom_config[i] != 0:
            for j in range(i + 1):
                if i == j and j == 0:
                    sigma[i] = (atom_config[i] - 1) * 0.3
                elif i == j and j != 0:
                    sigma[i] += (atom_config[i] - 1) * 0.35
                elif j in [3, 5, 6, 8, 9]:
                    sigma[i] += atom_config[j] * 1
                elif i - j == 1:
                    sigma[i] += atom_config[j] * 0.85
                elif i - j > 1:
                    sigma[i] += atom_config[j] * 1
        sigma[i] = atom.GetAtomicNum() - sigma[i]
    for i in range(len(atom_config)):
        e += -13.6 * ((sigma[i] / n_star[i]) ** 2)
    return -np.log(-e)

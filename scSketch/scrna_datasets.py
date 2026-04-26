import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import scanpy as sc
import os
import pickle
import numpy as np
import h5py
import tempfile
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from .progeny_utils import PROGENyCalculator


def _sanitize_h5ad_compat(source_path):
    source_path = Path(source_path)
    cache_dir = Path(tempfile.gettempdir()) / "scSketch_h5ad_compat"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sanitized_path = cache_dir / source_path.name

    if sanitized_path.exists():
        return sanitized_path

    with h5py.File(source_path, "r") as source_file, h5py.File(sanitized_path, "w") as target_file:
        for key, value in source_file.attrs.items():
            target_file.attrs[key] = value

        for key in source_file.keys():
            source_file.copy(key, target_file, name=key)

        if "uns" in target_file and "log1p" in target_file["uns"] and "base" in target_file["uns"]["log1p"]:
            del target_file["uns"]["log1p"]["base"]

    return sanitized_path


def read_h5ad_compat(path):
    try:
        return sc.read_h5ad(path)
    except Exception as exc:
        compat_path = _sanitize_h5ad_compat(path)
        print(f"INFO: using compatibility copy for {Path(path).name}: {compat_path}")
        try:
            return sc.read_h5ad(compat_path)
        except Exception:
            raise exc

def Drug_dose_encoder(drug_SMILES_list: list, dose_list: list, num_Bits=1024, comb_num=1):
    """
    adopted from PRnet @Author: Zijian Yuan
    Encode SMILES of drug to rFCFP fingerprint
    """
    drug_len = len(drug_SMILES_list)
    fcfp4_array = np.zeros((drug_len, num_Bits))

    if comb_num==1:
        for i, smiles in enumerate(drug_SMILES_list):
            smi = smiles
            mol = Chem.MolFromSmiles(smi)
            fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_Bits).ToBitString()
            fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
            fcfp4_list = fcfp4_list*np.log10(dose_list[i]+1)
            fcfp4_array[i] = fcfp4_list
    else:
        for i, smiles in enumerate(drug_SMILES_list):
            smiles_list = smiles.split('+')
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_Bits).ToBitString()
                fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
                fcfp4_list = fcfp4_list*np.log10(float(dose_list[i])+1)
                fcfp4_array[i] += fcfp4_list
    return fcfp4_array 

class AnnDataDataset(Dataset):
    def __init__(self, adata, control_adata=None, use_drug_structure=False, comb_num=1, 
                 progeny_sketches=None):
        """
        Args:
            adata: AnnData object with treated cells
            control_adata: AnnData object with control cells (if use_drug_structure)
            use_drug_structure: Whether to use drug structure information
            comb_num: Number of drug combinations
            progeny_sketches: Pre-computed PROGENy sketches (B, 14) tensor, or None
        """
        self.use_drug_structure = use_drug_structure
        if type(adata.X)==np.ndarray:
            self.features = torch.tensor(adata.X, dtype=torch.float32)
        else:
            self.features = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        
        # Store PROGENy sketches if provided
        self.progeny_sketches = progeny_sketches
        
        if self.use_drug_structure:
            if type(control_adata.X)==np.ndarray:
                self.control_features = torch.tensor(control_adata.X, dtype=torch.float32)
            else:
                self.control_features = torch.tensor(control_adata.X.toarray(), dtype=torch.float32)
                
            self.drug_type_list = adata.obs['SMILES'].to_list()
            self.dose_list = adata.obs['dose'].to_list()
            #self.encoded_obs_tensor = torch.tensor(adata.obs['Group'].copy().values, dtype=torch.float32)
            self.encoded_obs_tensor = adata.obs['Group'].copy().values
            self.encode_drug_doses = Drug_dose_encoder(self.drug_type_list, self.dose_list, comb_num=comb_num)
            self.encode_drug_doses = torch.tensor(self.encode_drug_doses, dtype=torch.float32)
        else:
            self.encoded_obs_tensor = adata.obs['Group'].copy().values
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
       
        if self.use_drug_structure:
            batch_dict = {
                'feature': self.features[idx], 
                'drug_dose': self.encode_drug_doses[idx], 
                'group': self.encoded_obs_tensor[idx],
                'control_feature': self.control_features[idx]
            }
            # Add PROGENy sketch if available
            if self.progeny_sketches is not None:
                batch_dict['progeny_sketch'] = self.progeny_sketches[idx]
            return batch_dict
        else:
            return {'feature':self.features[idx], 'group': self.encoded_obs_tensor[idx]}
            
    

def prepared_data(data_dir=None, control_data_dir=None, batch_size=64, use_drug_structure=False, 
                 comb_num=1, progeny_model_path=None, progeny_organism='human'):
    """
    Prepare data loader with optional PROGENy sketch computation.
    
    Args:
        data_dir: Path to treated cell data (h5ad)
        control_data_dir: Path to control cell data (h5ad)
        batch_size: Batch size
        use_drug_structure: Whether to use drug structure
        comb_num: Number of drug combinations
        progeny_model_path: Path to PROGENy model matrix
        progeny_organism: 'human' or 'mouse'
    
    Returns:
        dataloader: DataLoader object
        sketch_info: Dict with 'type' ('progeny' or 'marker') and 'data' (sketches or indices)
    """
    
    train_adata = read_h5ad_compat(data_dir)
    if use_drug_structure:
        control_adata = read_h5ad_compat(control_data_dir)
    else:
        control_adata = None
    
    progeny_sketches = None
    sketch_info = {'type': None, 'data': None}
    
    # Compute PROGENy sketches (always enabled when drug structure is used)
    if use_drug_structure:
        print("Computing PROGENy pathway activity sketches...")
        
        # Get gene names from adata first
        if hasattr(train_adata, 'var_names'):
            gene_names = list(train_adata.var_names)
            print(f"DEBUG: Loaded {len(gene_names)} gene names from adata")
        else:
            gene_names = None
            print("Warning: No gene names found in adata. Assuming genes match PROGENy model order.")
        
        # Initialize PROGENy calculator with gene list
        print("DEBUG: Initializing PROGENy calculator...")
        progeny_calc = PROGENyCalculator(
            model_matrix_path=progeny_model_path,
            organism=progeny_organism,
            top_genes=None,
            device='cpu',  # Compute on CPU during data loading
            gene_list=gene_names  # Pass gene names for simplified model
        )
        print("DEBUG: PROGENy calculator initialized")
        
        # Get expression data
        if type(train_adata.X) == np.ndarray:
            treated_expr = train_adata.X
        else:
            treated_expr = train_adata.X.toarray()
        
        if type(control_adata.X) == np.ndarray:
            control_expr = control_adata.X
        else:
            control_expr = control_adata.X.toarray()
        
        # Compute PROGENy sketches: s = f_PROGENy(x_treated) - f_PROGENy(x_control)
        # This is the pathway activity change (Δ)
        print(f"DEBUG: Computing pathway delta for {treated_expr.shape[0]} samples...")
        progeny_sketches = progeny_calc.compute_pathway_delta(
            expression_treated=treated_expr,
            expression_control=control_expr,
            gene_names=gene_names,
            normalize=True
        )
        print("DEBUG: Pathway delta computed")
        
        print(f"PROGENy sketches computed: shape {progeny_sketches.shape}")
        print(f"Pathway names: {progeny_calc.pathway_names}")
        print(f"Sketch statistics: mean={progeny_sketches.mean(dim=0)}, std={progeny_sketches.std(dim=0)}")
        
        sketch_info = {
            'type': 'progeny',
            'data': progeny_sketches,
            'pathway_names': progeny_calc.pathway_names,
            'n_pathways': len(progeny_calc.pathway_names),
            'progeny_calculator': progeny_calc  # Add calculator for consistency loss
        }
    
    # Create dataset with PROGENy sketches
    _data_dataset = AnnDataDataset(
        train_adata, 
        control_adata, 
        use_drug_structure, 
        comb_num,
        progeny_sketches=progeny_sketches
    )

    dataloader = DataLoader(
                _data_dataset, 
                batch_size=batch_size,
                shuffle=True, 
                )
        
    return dataloader, sketch_info
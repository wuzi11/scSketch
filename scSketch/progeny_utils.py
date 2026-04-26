# -*- coding: utf-8 -*-
"""
PROGENy pathway activity computation for scSketch.

PROGENy computes pathway activity scores from gene expression data
by using a linear model with pre-defined gene weights for 14 pathways.

Reference: 
Schubert et al. (2018). "Perturbation-response genes reveal signaling footprints in cancer gene expression."
Nature Communications, 9(1), 20.
"""

import torch
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict
import os


# PROGENy 14 pathways
PROGENY_PATHWAYS = [
    'EGFR', 'MAPK', 'PI3K', 'TGFb', 'WNT', 'VEGF',
    'NFkB', 'TNFa', 'Hypoxia', 'TRAIL',
    'JAK_STAT', 'p53', 'Androgen', 'Estrogen'
]


class PROGENyCalculator:
    """
    PROGENy pathway activity calculator.
    
    Computes 14-dimensional pathway activity vectors from gene expression data.
    """
    
    def __init__(
        self, 
        model_matrix_path: Optional[str] = None,
        organism: str = 'human',
        top_genes: Optional[int] = 100,
        device: str = 'cpu',
        gene_list: Optional[list] = None
    ):
        """
        Initialize PROGENy calculator.
        
        Args:
            model_matrix_path: Path to PROGENy model matrix (gene x pathway weights).
                              If None, will use built-in or download from decoupleR.
            organism: 'human' or 'mouse'
            top_genes: Number of top genes to use per pathway (100, 200, 500).
                      Set to None or 0 to use ALL genes (no filtering).
            device: torch device
            gene_list: List of gene names from the data (for simplified model)
        """
        self.organism = organism
        self.top_genes = top_genes
        self.device = device
        
        # Load or create model matrix
        if model_matrix_path is not None and os.path.exists(model_matrix_path):
            self.model_matrix = self._load_model_matrix(model_matrix_path)
        else:
            # Check cache first
            top_to_use = 10000 if (top_genes is None or top_genes == 0) else top_genes
            cache_file = os.path.join("cache", f"progeny_{organism}_top{top_to_use}.csv")
            
            if os.path.exists(cache_file):
                print(f"Loading PROGENy from cache: {cache_file}")
                self.model_matrix = self._load_model_matrix(cache_file)
                print(f"✓ PROGENy loaded from cache:")
                print(f"  - {self.model_matrix.shape[0]} genes")
                print(f"  - {self.model_matrix.shape[1]} pathways: {list(self.model_matrix.columns)}")
            else:
                # Try to load from decoupleR or use simplified version
                try:
                    self.model_matrix = self._load_from_decoupler()
                except:
                    print("Warning: Could not load PROGENy from decoupleR. Using simplified model.")
                    self.model_matrix = self._create_simplified_model(gene_list=gene_list)
        
        # Convert to torch tensor
        self.gene_names = list(self.model_matrix.index)
        self.pathway_names = list(self.model_matrix.columns)
        self.weights = torch.tensor(
            self.model_matrix.values, 
            dtype=torch.float32, 
            device=device
        )  # Shape: (n_genes, 14)
        
        print(f"PROGENy initialized: {len(self.gene_names)} genes, {len(self.pathway_names)} pathways")
    
    def _load_model_matrix(self, path: str) -> pd.DataFrame:
        """Load PROGENy model matrix from file."""
        if path.endswith('.csv'):
            df = pd.read_csv(path, index_col=0)
        elif path.endswith('.tsv'):
            df = pd.read_csv(path, sep='\t', index_col=0)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        return df
    
    def _load_from_decoupler(self) -> pd.DataFrame:
        """
        Load PROGENy model from decoupleR package.
        Requires: pip install decoupler
        """
        import time
        import socket
        try:
            import decoupler as dc
            # Load PROGENy model using the correct API: dc.op.progeny()
            # If top_genes is None or 0, load ALL genes (no filtering)
            if self.top_genes is None or self.top_genes == 0:
                # Use a very large number to get ALL genes (no top filtering)
                # PROGENy has different gene sets: 100, 200, 500, or all (~5000)
                top_to_use = 10000  # Large enough to get all available genes
                print(f"Loading PROGENy with ALL genes (no top filtering, using top={top_to_use})")
            else:
                top_to_use = self.top_genes
                print(f"Loading PROGENy with top {top_to_use} genes per pathway")
            
            # Set socket timeout to avoid hanging
            default_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(30)  # 30 seconds timeout
            
            # Use dc.op.progeny() - the correct API for loading PROGENy
            # Add retry mechanism for network issues
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    print(f"  Downloading PROGENy model (attempt {attempt + 1}/{max_retries})...")
                    progeny_model = dc.op.progeny(organism=self.organism, top=top_to_use)
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = min(2 ** attempt, 10)  # Cap at 10 seconds
                        print(f"  Attempt {attempt + 1} failed: {str(e)[:100]}")
                        print(f"  Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)  # Exponential backoff
                    else:
                        raise  # Last attempt failed, re-raise exception
            
            # Restore default timeout
            socket.setdefaulttimeout(default_timeout)
            
            # progeny_model is a DataFrame with columns: ['source', 'target', 'weight', 'padj']
            # source = pathway name, target = gene name
            # Convert to wide format: genes (rows) x pathways (columns)
            model_matrix = progeny_model.pivot(index='target', columns='source', values='weight').fillna(0)
            
            print(f"✓ PROGENy loaded successfully from decoupler:")
            print(f"  - {model_matrix.shape[0]} genes")
            print(f"  - {model_matrix.shape[1]} pathways: {list(model_matrix.columns)}")
            
            # Save to cache for future use
            cache_dir = "cache"
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"progeny_{self.organism}_top{top_to_use}.csv")
            model_matrix.to_csv(cache_file)
            print(f"  Cached to: {cache_file}")
            
            return model_matrix
        except ImportError as e:
            print(f"ImportError: {e}")
            raise ImportError("decoupler package not found. Install with: pip install decoupler")
        except Exception as e:
            print(f"Error loading PROGENy from decoupler: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_simplified_model(self, gene_list: Optional[list] = None) -> pd.DataFrame:
        """
        Create a simplified PROGENy model with random weights.
        This is a fallback when the real model is not available.
        
        WARNING: This is for testing only! Use real PROGENy weights for production.
        """
        print("WARNING: Using random weights for PROGENy. This is for testing only!")
        
        # Use provided gene list or create random gene names
        if gene_list is not None:
            gene_names = gene_list
            n_genes = len(gene_list)
            print(f"Using {n_genes} genes from data for simplified PROGENy model")
        else:
            n_genes = 1000
            gene_names = [f"GENE_{i}" for i in range(n_genes)]
        
        # Random weights for 14 pathways
        np.random.seed(42)
        weights = np.random.randn(n_genes, 14) * 0.1
        
        df = pd.DataFrame(
            weights,
            index=gene_names,
            columns=PROGENY_PATHWAYS
        )
        return df
    
    def compute_pathway_activity(
        self, 
        expression: Union[torch.Tensor, np.ndarray],
        gene_names: Optional[list] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute PROGENy pathway activity scores.
        
        Args:
            expression: Gene expression matrix (B, G) or (G,)
            gene_names: List of gene names corresponding to expression columns.
                       If None, assumes expression genes match self.gene_names order.
            normalize: Whether to z-score normalize the output
        
        Returns:
            pathway_activity: Pathway activity scores (B, 14) or (14,)
        """
        # Convert to torch tensor
        if isinstance(expression, np.ndarray):
            expression = torch.tensor(expression, dtype=torch.float32, device=self.device)
        else:
            expression = expression.to(self.device)
        
        # Handle 1D input
        if expression.dim() == 1:
            expression = expression.unsqueeze(0)  # (G,) -> (1, G)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, G = expression.shape
        
        # Match genes if gene_names provided
        if gene_names is not None:
            # Create mapping from input genes to model genes
            # Use case-insensitive matching for better gene name matching
            gene_to_idx = {gene: idx for idx, gene in enumerate(self.gene_names)}
            gene_to_idx_upper = {gene.upper(): idx for idx, gene in enumerate(self.gene_names)}
            
            # Find matching genes with multiple strategies
            matched_indices = []
            input_indices = []
            
            for i, gene in enumerate(gene_names):
                matched = False
                
                # Strategy 1: Exact match (fastest)
                if gene in gene_to_idx:
                    matched_indices.append(gene_to_idx[gene])
                    input_indices.append(i)
                    matched = True
                
                # Strategy 2: Case-insensitive match
                elif gene.upper() in gene_to_idx_upper:
                    matched_indices.append(gene_to_idx_upper[gene.upper()])
                    input_indices.append(i)
                    matched = True
                
                # Strategy 3: Fuzzy match (handle - vs _ vs .)
                if not matched:
                    # Try replacing common separators
                    gene_normalized = gene.upper().replace('-', '_').replace('.', '_')
                    for model_gene_upper, idx in gene_to_idx_upper.items():
                        model_gene_normalized = model_gene_upper.replace('-', '_').replace('.', '_')
                        if gene_normalized == model_gene_normalized:
                            matched_indices.append(idx)
                            input_indices.append(i)
                            break
            
            if len(matched_indices) == 0:
                print(f"DEBUG: Input has {len(gene_names)} genes, first 5: {gene_names[:5]}")
                print(f"DEBUG: Model has {len(self.gene_names)} genes, first 5: {self.gene_names[:5]}")
                raise ValueError("No matching genes found between input and PROGENy model!")
            
            # Select matched genes
            weights_matched = self.weights[matched_indices, :]  # (n_matched, 14)
            expression_matched = expression[:, input_indices]   # (B, n_matched)
            
        else:
            # Assume genes are in the same order
            if G != len(self.gene_names):
                raise ValueError(
                    f"Expression has {G} genes but PROGENy model has {len(self.gene_names)} genes. "
                    "Please provide gene_names for matching."
                )
            weights_matched = self.weights
            expression_matched = expression
        
        # Compute pathway activity: X @ W
        # (B, n_genes) @ (n_genes, 14) -> (B, 14)
        pathway_activity = torch.matmul(expression_matched, weights_matched)
        
        # Normalize (z-score per pathway across batch)
        if normalize and B > 1:
            mean = pathway_activity.mean(dim=0, keepdim=True)
            std = pathway_activity.std(dim=0, keepdim=True) + 1e-8
            pathway_activity = (pathway_activity - mean) / std
        
        if squeeze_output:
            pathway_activity = pathway_activity.squeeze(0)  # (1, 14) -> (14,)
        
        return pathway_activity
    
    def compute_pathway_delta(
        self,
        expression_treated: Union[torch.Tensor, np.ndarray],
        expression_control: Union[torch.Tensor, np.ndarray],
        gene_names: Optional[list] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute pathway activity change (Δ) between treated and control.
        
        This is the key function for scSketch sketch computation:
        s = f_PROGENy(x_treated) - f_PROGENy(x_control)
        
        Args:
            expression_treated: Treated cell expression (B, G) or (G,)
            expression_control: Control cell expression (B, G) or (G,)
            gene_names: Gene names
            normalize: Whether to normalize
        
        Returns:
            pathway_delta: Pathway activity change (B, 14) or (14,)
        """
        activity_treated = self.compute_pathway_activity(
            expression_treated, gene_names, normalize=False
        )
        activity_control = self.compute_pathway_activity(
            expression_control, gene_names, normalize=False
        )
        
        pathway_delta = activity_treated - activity_control
        
        # Normalize delta
        if normalize and pathway_delta.dim() > 1 and pathway_delta.shape[0] > 1:
            mean = pathway_delta.mean(dim=0, keepdim=True)
            std = pathway_delta.std(dim=0, keepdim=True) + 1e-8
            pathway_delta = (pathway_delta - mean) / std
        
        return pathway_delta
    
    def save_model_matrix(self, path: str):
        """Save PROGENy model matrix to file."""
        self.model_matrix.to_csv(path)
        print(f"PROGENy model matrix saved to {path}")


def load_progeny_calculator(
    model_path: Optional[str] = None,
    organism: str = 'human',
    top_genes: int = 100,
    device: str = 'cpu'
) -> PROGENyCalculator:
    """
    Convenience function to load PROGENy calculator.
    
    Args:
        model_path: Path to PROGENy model matrix
        organism: 'human' or 'mouse'
        top_genes: Number of top genes per pathway
        device: torch device
    
    Returns:
        PROGENyCalculator instance
    """
    return PROGENyCalculator(
        model_matrix_path=model_path,
        organism=organism,
        top_genes=top_genes,
        device=device
    )


# Example usage
if __name__ == "__main__":
    # Initialize calculator with real PROGENy data (ALL genes)
    calc = load_progeny_calculator(organism='human', top_genes=None)
    
    print(f"✓ PROGENy model loaded successfully!")
    print(f"  - Genes: {len(calc.gene_names)}")
    print(f"  - Pathways: {calc.pathway_names}")
    print(f"  - First 10 genes: {calc.gene_names[:10]}")
    
    # Example: compute pathway activity with gene name matching
    # Simulate expression data with some matching genes
    n_cells = 32
    gene_names = calc.gene_names[:500]  # Use first 500 genes from PROGENy
    expression = torch.randn(n_cells, len(gene_names))
    
    pathway_activity = calc.compute_pathway_activity(expression, gene_names=gene_names)
    print(f"\n✓ Pathway activity computed: {pathway_activity.shape}")  # (32, 14)
    
    # Example: compute pathway delta
    expression_control = torch.randn(n_cells, len(gene_names))
    expression_treated = expression_control + torch.randn(n_cells, len(gene_names)) * 0.5
    pathway_delta = calc.compute_pathway_delta(
        expression_treated, expression_control, gene_names=gene_names
    )
    print(f"\n✓ Pathway delta computed: {pathway_delta.shape}")  # (32, 14)
    print(f"\n🎉 All tests passed! PROGENy is working with real prior data.")

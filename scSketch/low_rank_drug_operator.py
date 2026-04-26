# -*- coding: utf-8 -*-
"""
Low-Rank Drug Operator Module

This module implements a low-rank drug perturbation operator that acts on cell states.
The key idea is that drugs don't affect the entire state space, but only a few 
regulatory pathways/axes.

Mathematical formulation:
    z_drug = z + U_d @ V_d^T @ z
    
where:
    - z: cell state (B, d)
    - e_d: drug embedding (B, drug_dim)
    - U_d, V_d: low-rank matrices (d, r) where r << d
    - U_d = f_U(e_d), V_d = f_V(e_d)
    - V_d^T @ z: projects cell state to drug action subspace (r dimensions)
    - U_d @ (...): maps perturbation back to full state space (d dimensions)

Biological interpretation:
    - Drugs act on specific pathways (low-rank structure)
    - r: number of affected regulatory axes (e.g., 4-16)
    - The operator is conditioned on drug identity and dose
"""

import torch
import torch.nn as nn
from typing import Optional


class LowRankDrugOperator(nn.Module):
    """
    Low-rank drug operator that applies drug perturbations to cell states.
    
    The operator learns two low-rank matrices U_d and V_d from drug embeddings,
    and applies an additive low-rank perturbation: z_drug = z + U_d @ V_d^T @ z
    """
    
    def __init__(
        self,
        state_dim: int,
        drug_dim: int,
        rank: int = 8,
        use_mlp: bool = True,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize Low-Rank Drug Operator.
        
        Args:
            state_dim: Dimension of cell state (d)
            drug_dim: Dimension of drug embedding
            rank: Rank of the low-rank matrices (r), should be << state_dim
            use_mlp: If True, use MLP for f_U and f_V; otherwise use linear layers
            hidden_dim: Hidden dimension for MLP (default: drug_dim)
            dropout: Dropout rate for MLP
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.drug_dim = drug_dim
        self.rank = rank
        self.use_mlp = use_mlp
        
        if hidden_dim is None:
            hidden_dim = drug_dim
        
        # Verify rank constraint
        if rank >= state_dim:
            raise ValueError(f"Rank {rank} should be much smaller than state_dim {state_dim}")
        
        # f_U: drug embedding -> U_d (state_dim, rank)
        if use_mlp:
            self.f_U = nn.Sequential(
                nn.Linear(drug_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, state_dim * rank),
            )
        else:
            self.f_U = nn.Linear(drug_dim, state_dim * rank)
        
        # f_V: drug embedding -> V_d (state_dim, rank)
        if use_mlp:
            self.f_V = nn.Sequential(
                nn.Linear(drug_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, state_dim * rank),
            )
        else:
            self.f_V = nn.Linear(drug_dim, state_dim * rank)
        
        # Optional: scale factor for perturbation magnitude
        self.scale = nn.Parameter(torch.ones(1))
        
        print(f"LowRankDrugOperator initialized:")
        print(f"  - State dimension: {state_dim}")
        print(f"  - Drug dimension: {drug_dim}")
        print(f"  - Rank: {rank} (compression ratio: {rank/state_dim:.2%})")
        print(f"  - Use MLP: {use_mlp}")
    
    def forward(
        self,
        z: torch.Tensor,
        drug_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply low-rank drug perturbation to cell state.
        
        Args:
            z: Cell state (B, state_dim)
            drug_emb: Drug embedding (B, drug_dim)
        
        Returns:
            z_drug: Perturbed cell state (B, state_dim)
        """
        B = z.shape[0]
        
        # Generate U_d and V_d from drug embedding
        U_d = self.f_U(drug_emb)  # (B, state_dim * rank)
        V_d = self.f_V(drug_emb)  # (B, state_dim * rank)
        
        # Reshape to (B, state_dim, rank)
        U_d = U_d.view(B, self.state_dim, self.rank)
        V_d = V_d.view(B, self.state_dim, self.rank)
        
        # Apply low-rank perturbation: z_drug = z + U_d @ V_d^T @ z
        # Step 1: V_d^T @ z -> project to drug action subspace
        # V_d: (B, state_dim, rank), z: (B, state_dim) -> (B, state_dim, 1)
        z_expanded = z.unsqueeze(-1)  # (B, state_dim, 1)
        
        # V_d^T @ z: (B, rank, state_dim) @ (B, state_dim, 1) -> (B, rank, 1)
        V_d_T = V_d.transpose(1, 2)  # (B, rank, state_dim)
        projection = torch.bmm(V_d_T, z_expanded)  # (B, rank, 1)
        
        # Step 2: U_d @ projection -> map back to full state space
        # U_d: (B, state_dim, rank) @ (B, rank, 1) -> (B, state_dim, 1)
        perturbation = torch.bmm(U_d, projection)  # (B, state_dim, 1)
        perturbation = perturbation.squeeze(-1)  # (B, state_dim)
        
        # Apply scaled perturbation
        z_drug = z + self.scale * perturbation
        
        return z_drug
    
    def get_perturbation_magnitude(
        self,
        z: torch.Tensor,
        drug_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the magnitude of drug perturbation without applying it.
        Useful for analysis and visualization.
        
        Args:
            z: Cell state (B, state_dim)
            drug_emb: Drug embedding (B, drug_dim)
        
        Returns:
            magnitude: L2 norm of perturbation (B,)
        """
        B = z.shape[0]
        
        U_d = self.f_U(drug_emb).view(B, self.state_dim, self.rank)
        V_d = self.f_V(drug_emb).view(B, self.state_dim, self.rank)
        
        z_expanded = z.unsqueeze(-1)
        V_d_T = V_d.transpose(1, 2)
        projection = torch.bmm(V_d_T, z_expanded)
        perturbation = torch.bmm(U_d, projection).squeeze(-1)
        
        magnitude = torch.norm(self.scale * perturbation, dim=1)
        return magnitude
    
    def get_drug_subspace(
        self,
        drug_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the drug action subspace matrices U_d and V_d.
        Useful for interpretability analysis.
        
        Args:
            drug_emb: Drug embedding (B, drug_dim)
        
        Returns:
            U_d: (B, state_dim, rank)
            V_d: (B, state_dim, rank)
        """
        B = drug_emb.shape[0]
        
        U_d = self.f_U(drug_emb).view(B, self.state_dim, self.rank)
        V_d = self.f_V(drug_emb).view(B, self.state_dim, self.rank)
        
        return U_d, V_d


class ConditionalLowRankDrugOperator(nn.Module):
    """
    Conditional low-rank drug operator that can be applied at different stages.
    
    This variant allows the operator to be conditioned on additional information
    such as cell type, time step, or other contextual features.
    """
    
    def __init__(
        self,
        state_dim: int,
        drug_dim: int,
        condition_dim: int = 0,
        rank: int = 8,
        use_mlp: bool = True,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize Conditional Low-Rank Drug Operator.
        
        Args:
            state_dim: Dimension of cell state (d)
            drug_dim: Dimension of drug embedding
            condition_dim: Dimension of additional conditioning (e.g., time embedding)
            rank: Rank of the low-rank matrices (r)
            use_mlp: If True, use MLP for f_U and f_V
            hidden_dim: Hidden dimension for MLP
            dropout: Dropout rate
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.drug_dim = drug_dim
        self.condition_dim = condition_dim
        self.rank = rank
        
        # Combine drug and condition embeddings
        input_dim = drug_dim + condition_dim
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        # f_U and f_V now take combined input
        if use_mlp:
            self.f_U = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, state_dim * rank),
            )
            self.f_V = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, state_dim * rank),
            )
        else:
            self.f_U = nn.Linear(input_dim, state_dim * rank)
            self.f_V = nn.Linear(input_dim, state_dim * rank)
        
        self.scale = nn.Parameter(torch.ones(1))
        
        print(f"ConditionalLowRankDrugOperator initialized:")
        print(f"  - State dimension: {state_dim}")
        print(f"  - Drug dimension: {drug_dim}")
        print(f"  - Condition dimension: {condition_dim}")
        print(f"  - Rank: {rank}")
    
    def forward(
        self,
        z: torch.Tensor,
        drug_emb: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply conditional low-rank drug perturbation.
        
        Args:
            z: Cell state (B, state_dim)
            drug_emb: Drug embedding (B, drug_dim)
            condition: Additional conditioning (B, condition_dim) or None
        
        Returns:
            z_drug: Perturbed cell state (B, state_dim)
        """
        B = z.shape[0]
        
        # Combine drug and condition embeddings
        if condition is not None and self.condition_dim > 0:
            combined_emb = torch.cat([drug_emb, condition], dim=-1)
        else:
            # When no condition, pad with zeros to match expected input size
            if self.condition_dim > 0:
                zero_condition = torch.zeros(B, self.condition_dim, device=drug_emb.device, dtype=drug_emb.dtype)
                combined_emb = torch.cat([drug_emb, zero_condition], dim=-1)
            else:
                combined_emb = drug_emb
        
        # Generate U_d and V_d
        U_d = self.f_U(combined_emb).view(B, self.state_dim, self.rank)
        V_d = self.f_V(combined_emb).view(B, self.state_dim, self.rank)
        
        # Apply low-rank perturbation
        z_expanded = z.unsqueeze(-1)
        V_d_T = V_d.transpose(1, 2)
        projection = torch.bmm(V_d_T, z_expanded)
        perturbation = torch.bmm(U_d, projection).squeeze(-1)
        
        z_drug = z + self.scale * perturbation
        
        return z_drug

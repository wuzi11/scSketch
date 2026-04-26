# -*- coding: utf-8 -*-
"""
Stage 1: Pathway Change Predictor (g function)

This module predicts pathway-level changes (Δs) from control expression and drug information.

Input:
    - x_control: Control cell expression (B, G)
    - drug: Drug embedding
    - dose: Drug dose

Output:
    - Δs_hat: Predicted pathway change (B, 14) for 14 PROGENy pathways

Supervision:
    - Δs_true = PROGENy(x_treated) - PROGENy(x_control)

Loss:
    - L_path = ||Δs_hat - Δs_true||^2
"""

import torch
import torch.nn as nn
from typing import Optional


class PathwayPredictor(nn.Module):
    """
    Stage 1: Pathway change predictor g(x_control, drug, dose) → Δs_hat
    
    This module predicts the 14-dimensional PROGENy pathway activity changes
    from control cell expression and drug/dose information.
    """
    
    def __init__(
        self,
        gene_size: int,
        hidden_size: int = 512,
        n_pathways: int = 14,
        use_drug_structure: bool = True,
        drug_dimension: int = 1024,
        dropout: float = 0.1,
    ):
        """
        Initialize pathway predictor.
        
        Args:
            gene_size: Number of genes in input
            hidden_size: Hidden layer size
            n_pathways: Number of pathways to predict (14 for PROGENy)
            use_drug_structure: Whether to use drug structure embeddings
            drug_dimension: Dimension of drug embeddings
            dropout: Dropout rate
        """
        super().__init__()
        
        self.gene_size = gene_size
        self.hidden_size = hidden_size
        self.n_pathways = n_pathways
        self.use_drug_structure = use_drug_structure
        self.drug_dimension = drug_dimension
        
        # Encode control expression
        self.control_encoder = nn.Sequential(
            nn.Linear(gene_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
        # Drug/dose encoder (if used)
        if use_drug_structure:
            self.drug_encoder = nn.Sequential(
                nn.Linear(drug_dimension, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
        else:
            self.drug_encoder = None
        
        # Fusion and prediction
        fusion_input_size = hidden_size * 2 if use_drug_structure else hidden_size
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_pathways),
        )
        
        print(f"PathwayPredictor initialized:")
        print(f"  - Gene size: {gene_size}")
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Pathways: {n_pathways}")
        print(f"  - Use drug structure: {use_drug_structure}")
    
    def forward(
        self,
        x_control: torch.Tensor,
        drug_dose: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict pathway changes.
        
        Args:
            x_control: Control cell expression (B, G)
            drug_dose: Drug/dose embedding (B, D)
        
        Returns:
            delta_s_hat: Predicted pathway change (B, 14)
        """
        # Encode control expression
        control_emb = self.control_encoder(x_control)  # (B, hidden)
        
        # Encode drug/dose if available
        if self.use_drug_structure and drug_dose is not None:
            drug_emb = self.drug_encoder(drug_dose)  # (B, hidden)
            # Concatenate control and drug embeddings
            fusion_input = torch.cat([control_emb, drug_emb], dim=-1)  # (B, 2*hidden)
        else:
            fusion_input = control_emb  # (B, hidden)
        
        # Predict pathway changes
        delta_s_hat = self.fusion_net(fusion_input)  # (B, 14)
        
        return delta_s_hat


def compute_pathway_loss(
    delta_s_hat: torch.Tensor,
    delta_s_true: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute pathway prediction loss.
    
    L_path = ||Δs_hat - Δs_true||^2
    
    Args:
        delta_s_hat: Predicted pathway change (B, 14)
        delta_s_true: Ground truth pathway change (B, 14)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        loss: Pathway prediction loss
    """
    if reduction == 'mean':
        return torch.nn.functional.mse_loss(delta_s_hat, delta_s_true, reduction='mean')
    elif reduction == 'sum':
        return torch.nn.functional.mse_loss(delta_s_hat, delta_s_true, reduction='sum')
    elif reduction == 'none':
        return torch.nn.functional.mse_loss(delta_s_hat, delta_s_true, reduction='none')
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


# Example usage
if __name__ == "__main__":
    # Test pathway predictor
    batch_size = 32
    gene_size = 2000
    drug_dim = 1024
    
    # Create predictor
    predictor = PathwayPredictor(
        gene_size=gene_size,
        hidden_size=512,
        n_pathways=14,
        use_drug_structure=True,
        drug_dimension=drug_dim,
    )
    
    # Test forward pass
    x_control = torch.randn(batch_size, gene_size)
    drug_dose = torch.randn(batch_size, drug_dim)
    
    delta_s_hat = predictor(x_control, drug_dose)
    print(f"\n✓ Pathway predictor output shape: {delta_s_hat.shape}")  # (32, 14)
    
    # Test loss
    delta_s_true = torch.randn(batch_size, 14)
    loss = compute_pathway_loss(delta_s_hat, delta_s_true)
    print(f"✓ Pathway loss: {loss.item():.4f}")
    
    print("\n🎉 PathwayPredictor test passed!")

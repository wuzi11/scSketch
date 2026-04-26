# -*- coding: utf-8 -*-
# @Author: Zijian Yuan
# DiT 1D Model for Single-cell RNA-seq Data using TabTransformer-style architecture

import torch
import torch.nn as nn
import numpy as np
import math
import sys
import os

# translated DiT translated
dit_path = os.path.join(os.path.dirname(__file__), '..', 'DiT-main', 'DiT-main')
if os.path.exists(dit_path):
    sys.path.insert(0, dit_path)
    try:
        from models import DiTBlock, TimestepEmbedder
        from timm.models.vision_transformer import Attention, Mlp
    except ImportError:
        DiTBlock = None
        TimestepEmbedder = None
        Attention = None
        Mlp = None
else:
    DiTBlock = None
    TimestepEmbedder = None
    Attention = None
    Mlp = None

from .precision import convert_module_to_f16, convert_module_to_f32
from .nn import timestep_embedding
from .pathway_predictor import PathwayPredictor
from .pos_embed import get_1d_sincos_pos_embed
from .low_rank_drug_operator import LowRankDrugOperator, ConditionalLowRankDrugOperator
from .encoder import EncoderMLPModel

def modulate(x, shift, scale):
    """Adaptive layer norm modulation"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_1d_sincos_pos_embed(embed_dim, length):
    """
    translated 1D translated/translated
    embed_dim: translated
    length: translated（translated）
    """
    assert embed_dim % 2 == 0
    
    # translated
    pos = np.arange(length, dtype=np.float32)
    
    # translated
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    
    # translated
    out = np.einsum('m,d->md', pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class GenePatchEmbed1D(nn.Module):
    """Convert 1D gene expression vectors to 1D patches.
    Unlike DiT's 2D grid approach, this keeps patches in 1D sequence.
    """
    def __init__(self, gene_size, patch_size, hidden_size, overlap=0, bias=True):
        super().__init__()
        self.gene_size = gene_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.overlap = overlap
        
        # Calculate stride and number of patches
        if overlap > 0:
            self.stride = patch_size - overlap
            num_patches = (gene_size - overlap + self.stride - 1) // self.stride
            if num_patches == 0:
                num_patches = 1
        else:
            self.stride = patch_size
            num_patches = (gene_size + patch_size - 1) // patch_size
        
        self.num_patches = num_patches
        
        # Linear projection from patch_size to hidden_size
        self.proj = nn.Linear(patch_size, hidden_size, bias=bias)
        
        # Register patch indices for unpatchify
        self._register_patch_indices()
    
    def _register_patch_indices(self):
        """Register mapping from patch index to original gene indices."""
        patch_indices = []
        for i in range(self.num_patches):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_size
            if start_idx >= self.gene_size:
                indices = []
            else:
                indices = list(range(start_idx, min(end_idx, self.gene_size)))
            patch_indices.append(indices)
        self.patch_indices = patch_indices
    
    def forward(self, x):
        """
        x: (B, gene_size)
        return: (B, num_patches, hidden_size)
        """
        B = x.shape[0]
        
        if self.overlap > 0:
            # Create overlapping patches
            patches = []
            for i in range(self.num_patches):
                start_idx = i * self.stride
                end_idx = start_idx + self.patch_size
                
                if start_idx >= self.gene_size:
                    patch = torch.zeros(B, self.patch_size, device=x.device, dtype=x.dtype)
                elif end_idx > self.gene_size:
                    patch = x[:, start_idx:self.gene_size]
                    pad_size = self.patch_size - patch.shape[1]
                    patch = torch.cat([patch, torch.zeros(B, pad_size, device=x.device, dtype=x.dtype)], dim=1)
                else:
                    patch = x[:, start_idx:end_idx]
                
                patches.append(patch)
            x = torch.stack(patches, dim=1)  # (B, num_patches, patch_size)
        else:
            # Non-overlapping patches
            pad_size = (self.patch_size - (x.shape[1] % self.patch_size)) % self.patch_size
            if pad_size > 0:
                x = torch.cat([x, torch.zeros(B, pad_size, device=x.device, dtype=x.dtype)], dim=1)
            x = x.reshape(B, -1, self.patch_size)  # (B, num_patches, patch_size)
        
        # Project to hidden dimension
        x = self.proj(x)  # (B, num_patches, hidden_size)
        return x
    
    def unpatchify(self, x):
        """Convert patches back to 1D gene vector.
        x: (B, num_patches, patch_size, out_channels)
        return: (B, gene_size, out_channels)
        """
        B = x.shape[0]
        out_channels = x.shape[-1]
        
        if self.overlap > 0:
            # Overlapping patches: use mean aggregation
            output = torch.zeros(B, self.gene_size, out_channels, device=x.device, dtype=x.dtype)
            count = torch.zeros(B, self.gene_size, device=x.device, dtype=x.dtype)
            
            for patch_idx in range(self.num_patches):
                gene_indices = self.patch_indices[patch_idx]
                if len(gene_indices) > 0:
                    patch_values = x[:, patch_idx, :len(gene_indices), :]
                    for i, gene_idx in enumerate(gene_indices):
                        if gene_idx < self.gene_size:
                            output[:, gene_idx, :] += patch_values[:, i, :]
                            count[:, gene_idx] += 1
            
            mask = count > 0
            output = output / count.unsqueeze(-1).clamp(min=1)
            output[~mask.unsqueeze(-1).expand_as(output)] = 0
        else:
            # Non-overlapping patches
            x = x.reshape(B, -1, out_channels)
            max_len = min(x.shape[1], self.gene_size)
            x = x[:, :max_len, :]
            
            if x.shape[1] < self.gene_size:
                pad_size = self.gene_size - x.shape[1]
                padding = torch.zeros(B, pad_size, out_channels, device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=1)
            output = x
        
        return output


class GeneEmbedding(nn.Module):
    """
    translated - translated TabTransformer translated
    translated token
    """
    def __init__(self, gene_size, hidden_size, use_value_embedding=True):
        super().__init__()
        self.gene_size = gene_size
        self.hidden_size = hidden_size
        self.use_value_embedding = use_value_embedding
        
        # translated（translated）
        self.column_embedding = nn.Embedding(gene_size, hidden_size)
        
        # translated（translated，translated）
        if use_value_embedding:
            # translated
            self.value_proj = nn.Linear(1, hidden_size)
        else:
            self.value_proj = None
    
    def forward(self, x):
        """
        x: (B, gene_size) - translated
        return: (B, gene_size, hidden_size)
        """
        B = x.shape[0]
        
        # translated (0, 1, 2, ..., gene_size-1)
        column_indices = torch.arange(self.gene_size, device=x.device).unsqueeze(0).expand(B, -1)
        
        # translated: (B, gene_size, hidden_size)
        col_emb = self.column_embedding(column_indices)
        
        if self.use_value_embedding and self.value_proj is not None:
            # translated: translated
            # x: (B, gene_size) -> (B, gene_size, 1) -> (B, gene_size, hidden_size)
            val_emb = self.value_proj(x.unsqueeze(-1))
            # translated
            x_emb = col_emb + val_emb
        else:
            # translated
            x_emb = col_emb
        
        return x_emb


class ConditionalLabelEmbedder(nn.Module):
    """
    translated，translated（translated DiTModel translated）
    """
    def __init__(self, hidden_size, num_classes=None, dropout_prob=0.1, 
                 use_drug_structure=False, drug_dimension=1024, latent_dim=60):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.use_drug_structure = use_drug_structure
        self.drug_dimension = drug_dimension
        self.latent_dim = latent_dim
        
        # translated（translated）
        if num_classes is not None:
            use_cfg_embedding = dropout_prob > 0
            self.class_embedding = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
            self.dropout_prob = dropout_prob
        else:
            self.class_embedding = None
            self.dropout_prob = 0
        
        # translated（translated）
        if latent_dim > 0:
            self.latent_proj = nn.Linear(latent_dim, hidden_size)
        else:
            self.latent_proj = None
        
        # translated（translated）
        if use_drug_structure:
            self.drug_proj = nn.Linear(drug_dimension, hidden_size)
        else:
            self.drug_proj = None
        
        # translated
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, labels=None, z_sem=None, drug_dose=None, training=True, force_drop_ids=None):
        """translated"""
        embeddings = []
        
        # translated
        if self.class_embedding is not None and labels is not None:
            if training and self.dropout_prob > 0:
                if force_drop_ids is None:
                    drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
                else:
                    drop_ids = force_drop_ids == 1
                labels = torch.where(drop_ids, self.num_classes, labels)
            class_emb = self.class_embedding(labels)
            embeddings.append(class_emb)
        
        # translated（translated）
        if self.latent_proj is not None and z_sem is not None:
            latent_emb = self.latent_proj(z_sem)
            embeddings.append(latent_emb)
        
        # translated
        if self.drug_proj is not None and drug_dose is not None:
            drug_emb = self.drug_proj(drug_dose)
            embeddings.append(drug_emb)
        
        # translated
        if len(embeddings) > 0:
            combined = sum(embeddings) / len(embeddings)
            combined = self.fusion(combined)
        else:
            device = labels.device if labels is not None else (z_sem.device if z_sem is not None else (drug_dose.device if drug_dose is not None else torch.device('cpu')))
            combined = torch.zeros(embeddings[0].shape[0] if embeddings else 1, 
                                 self.hidden_size, device=device, dtype=embeddings[0].dtype if embeddings else torch.float32)
        
        return combined


class DiTBlock1D(nn.Module):
    """
    1D DiT block with adaptive layer norm zero (adaLN-Zero) conditioning
    translated 1D translated DiT block
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # translated Attention（translated）translated
        if Attention is not None:
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        else:
            # translated
            self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        
        if Mlp is not None:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden_dim),
                approx_gelu(),
                nn.Linear(mlp_hidden_dim, hidden_size)
            )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        x: (B, gene_size, hidden_size)
        c: (B, hidden_size) - translated
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-attention
        if isinstance(self.attn, nn.MultiheadAttention):
            # translated PyTorch translated MultiheadAttention
            attn_out, _ = self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa),
                modulate(self.norm1(x), shift_msa, scale_msa),
                modulate(self.norm1(x), shift_msa, scale_msa)
            )
            x = x + gate_msa.unsqueeze(1) * attn_out
        else:
            # translated timm translated Attention
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        
        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer1D(nn.Module):
    """
    1D translated
    translated output_dim translated output_dim*2
    """
    def __init__(self, hidden_size, gene_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # translated out_channels（translated）
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        x: (B, gene_size, hidden_size)
        c: (B, hidden_size)
        return: (B, gene_size, out_channels)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)  # (B, gene_size, out_channels)
        return x


class TimestepEmbedderLocal(nn.Module):
    """translated（translated）"""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiT1DModel(nn.Module):
    """
    translated TabTransformer translated 1D DiT translated
    translated 1D translated，translated 2D translated
    translated：
    1. patch_size > 1: translated1D patch embedding (translatedpatch_size=4)
    2. patch_size = 1: translatedper-gene embedding (TabTransformertranslated)
    """
    def __init__(
        self,
        gene_size,
        output_dim,
        num_layers=12,
        hidden_size=768,
        num_heads=12,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=None,
        learn_sigma=False,
        use_checkpoint=False,
        use_fp16=False,
        use_encoder=False,
        use_drug_structure=False,
        drug_dimension=1024,
        comb_num=1,
        latent_dim=60,
        use_value_embedding=True,
        patch_size=1,
        patch_overlap=0,
    ):
        super().__init__()
        
        self.gene_size = gene_size
        self.output_dim = output_dim
        self.use_encoder = use_encoder
        self.use_drug_structure = use_drug_structure
        self.num_classes = num_classes
        self.learn_sigma = learn_sigma
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.patch_overlap = 0
        self.use_patch = (patch_size > 1)
        self.use_two_stage = False
        self.pathway_hidden_size = 512
        self.use_consistency_loss = True
        
        # translated（translated）
        if use_encoder:
            self.encoder = EncoderMLPModel(
                gene_size, hidden_size, num_classes, 
                use_drug_structure, drug_dimension, comb_num, 
                output_size=latent_dim
            )
        
        # translated：patchtranslatedper-genetranslated
        if self.use_patch:
            # translated1D patch embedding
            self.x_embedder = GenePatchEmbed1D(gene_size, patch_size, hidden_size, overlap=patch_overlap, bias=True)
            num_tokens = self.x_embedder.num_patches
            self.gene_embedding = None
        else:
            # translatedper-gene embedding（translated TabTransformer）
            self.gene_embedding = GeneEmbedding(gene_size, hidden_size, use_value_embedding)
            num_tokens = gene_size
            self.x_embedder = None
        
        # 1D translated
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_size), requires_grad=False)
        self._init_pos_embed()
        
        # translated
        if TimestepEmbedder is not None:
            self.t_embedder = TimestepEmbedder(hidden_size)
        else:
            self.t_embedder = TimestepEmbedderLocal(hidden_size)
        
        # translated
        self.y_embedder = ConditionalLabelEmbedder(
            hidden_size, num_classes, class_dropout_prob,
            use_drug_structure, drug_dimension, latent_dim
        )
        
        # DiT blocks (1D)
        self.blocks = nn.ModuleList([
            DiTBlock1D(hidden_size, num_heads, mlp_ratio=mlp_ratio) 
            for _ in range(num_layers)
        ])
        
        # translated
        if self.use_patch:
            # Patchtranslated：translatedpatch_sizetranslated，translatedunpatchify
            out_channels = patch_size * (2 if learn_sigma else 1)
        else:
            # Per-genetranslated：translatedoutput_dim
            out_channels = output_dim * 2 if learn_sigma else output_dim
        self.final_layer = FinalLayer1D(hidden_size, num_tokens, out_channels)
        
        # translated
        self.n_pathways = 14  # PROGENy pathways
        self.pathway_predictor = None
        self.pathway_proj = None
        
        # Low-Rank Drug Operator
        # NOTE: translated Encoder translated，translated DiT translated
        # translated，translated
        self.drug_operator = None
        
        # translated DiT translated drug operator（translated），translated
        # if self.use_low_rank_drug_op and use_drug_structure:
        #     if drug_op_conditional:
        #         self.drug_operator = ConditionalLowRankDrugOperator(
        #             state_dim=hidden_size,
        #             drug_dim=drug_dimension,
        #             condition_dim=hidden_size,
        #             rank=drug_op_rank,
        #             use_mlp=drug_op_use_mlp,
        #         )
        #     else:
        #         self.drug_operator = LowRankDrugOperator(
        #             state_dim=hidden_size,
        #             drug_dim=drug_dimension,
        #             rank=drug_op_rank,
        #             use_mlp=drug_op_use_mlp,
        #         )
        #     print(f"Low-Rank Drug Operator enabled in DiT backend (conditional={drug_op_conditional})")
        # else:
        #     self.drug_operator = None
        
        # translated
        self.initialize_weights()
        
        # FP16 translated
        if use_fp16:
            self.convert_to_fp16()
    
    def _init_pos_embed(self):
        """translated 1D translated"""
        num_tokens = self.pos_embed.shape[1]
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            num_tokens
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
    def initialize_weights(self):
        """translated"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # translated
        if self.use_patch:
            # translatedpatch projection
            if hasattr(self.x_embedder, 'proj'):
                w = self.x_embedder.proj.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                nn.init.constant_(self.x_embedder.proj.bias, 0)
        else:
            # translated
            nn.init.normal_(self.gene_embedding.column_embedding.weight, std=0.02)
            if self.gene_embedding.value_proj is not None:
                nn.init.normal_(self.gene_embedding.value_proj.weight, std=0.02)
        
        # translated
        if self.y_embedder.class_embedding is not None:
            nn.init.normal_(self.y_embedder.class_embedding.weight, std=0.02)
        
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(self, x, timesteps=None, **model_kwargs):
        """
        translated（translated）
        x: (B, gene_size) - translated
        timesteps: (B,) - translated
        model_kwargs: translated x_start, group, drug_dose, control_feature translated
        
        Returns:
            x_out: translated (B, G)
            delta_s_hat: translated (B, 14) translated None
        """
        B = x.shape[0]
        
        # translated
        labels = model_kwargs.get('group', None)
        drug_dose = model_kwargs.get('drug_dose', None)
        control_feature = model_kwargs.get('control_feature', None)
        z_sem = None
        
        # Stage 1: translated（translated）translated sketch（translated/translated）
        delta_s_hat = None
        if 'delta_s_override' in model_kwargs:
            # translated sketch：translated / translated / translated sketch translated
            delta_s_hat = model_kwargs['delta_s_override']  # [B, 14]
        elif self.use_two_stage and self.pathway_predictor is not None:
            if control_feature is not None:
                # translated Δs_hat = g(x_control, drug, dose)
                delta_s_hat = self.pathway_predictor(
                    x_control=control_feature,
                    drug_dose=drug_dose
                )  # [B, 14]
            else:
                # Fallback: translated noisy x translated（translated，translated）
                x_start = model_kwargs.get('x_start', x)
                delta_s_hat = self.pathway_predictor(
                    x_control=x_start,
                    drug_dose=drug_dose
                )  # [B, 14]
        
        if self.use_encoder:
            if 'z_mod' in model_kwargs:
                z_sem = model_kwargs['z_mod']
            elif self.num_classes is None:
                x_start = model_kwargs.get('x_start', x)
                z_sem = self.encoder(
                    x_start, label=None, 
                    drug_dose=drug_dose, 
                    control_feature=control_feature
                )
            else:
                x_start = model_kwargs.get('x_start', x)
                z_sem = self.encoder(
                    x_start, label=labels,
                    drug_dose=drug_dose,
                    control_feature=control_feature
                )
        
        # translated: (B, gene_size) -> (B, num_tokens, hidden_size)
        if self.use_patch:
            # Patchtranslated: (B, gene_size) -> (B, num_patches, hidden_size)
            x_emb = self.x_embedder(x)
        else:
            # Per-genetranslated: (B, gene_size) -> (B, gene_size, hidden_size)
            x_emb = self.gene_embedding(x)
        
        # translated
        x_emb = x_emb + self.pos_embed
        
        # translated
        t_emb = self.t_embedder(timesteps)  # (B, hidden_size)
        
        # translated
        y_emb = self.y_embedder(
            labels=labels,
            z_sem=z_sem,
            drug_dose=drug_dose,
            training=self.training
        )  # (B, hidden_size)
        
        # Stage 2: diffusion condition excludes sketch/pathway injection.
        # Keep delta_s_hat prediction for pathway/sketch supervision only.
        c = t_emb + y_emb  # (B, hidden_size)
        
        # translated DiT blocks
        for block in self.blocks:
            x_emb = block(x_emb, c)  # (B, num_tokens, hidden_size)
        
        # NOTE: Low-Rank Drug Operator translated Encoder translated
        # translated DiT blocks translated，translated
        # translated，translated
        # if self.drug_operator is not None and drug_dose is not None:
        #     B, num_tokens, hidden_size = x_emb.shape
        #     x_emb_flat = x_emb.reshape(B * num_tokens, hidden_size)
        #     drug_dose_expanded = drug_dose.unsqueeze(1).expand(B, num_tokens, -1).reshape(B * num_tokens, -1)
        #     if self.drug_op_conditional:
        #         c_expanded = c.unsqueeze(1).expand(B, num_tokens, -1).reshape(B * num_tokens, hidden_size)
        #         x_emb_flat = self.drug_operator(x_emb_flat, drug_dose_expanded, c_expanded)
        #     else:
        #         x_emb_flat = self.drug_operator(x_emb_flat, drug_dose_expanded)
        #     x_emb = x_emb_flat.reshape(B, num_tokens, hidden_size)
        
        # translated
        x_out = self.final_layer(x_emb, c)  # (B, num_tokens, out_channels)
        
        # translated
        if self.use_patch:
            # Patchtranslated: unpatchifytranslated1Dtranslated
            # x_out: (B, num_patches, patch_size * pred_channels)
            pred_channels = 2 if self.learn_sigma else 1
            x_out = x_out.reshape(B, self.x_embedder.num_patches, self.patch_size, pred_channels)
            # unpatchify: (B, num_patches, patch_size, pred_channels) -> (B, gene_size, pred_channels)
            x_out = self.x_embedder.unpatchify(x_out)
            
            # translated
            if self.learn_sigma:
                # (B, gene_size, 2) -> (B, gene_size*2)
                B2, G2, C2 = x_out.shape
                assert G2 == self.gene_size and C2 == 2
                x_out = x_out.reshape(B2, G2 * 2)
            else:
                # (B, gene_size, 1) -> (B, gene_size)
                x_out = x_out[..., 0]
        else:
            # Per-genetranslated: translated
            # translated，translated output_dim == gene_size
            if self.learn_sigma:
                # (B, gene_size, output_dim*2) -> (B, output_dim*2)
                if self.gene_size == self.output_dim:
                    # translated
                    indices = torch.arange(self.output_dim, device=x_out.device)
                    x_mean = x_out[:, indices, :self.output_dim]
                    x_var = x_out[:, indices, self.output_dim:]
                    x_mean = x_mean[:, indices, indices]  # (B, output_dim)
                    x_var = x_var[:, indices, indices]  # (B, output_dim)
                    x_out = torch.cat([x_mean, x_var], dim=1)  # (B, output_dim*2)
                else:
                    # translated
                    x_mean = x_out[:, :, :self.output_dim].mean(dim=1)
                    x_var = x_out[:, :, self.output_dim:].mean(dim=1)
                    x_out = torch.cat([x_mean, x_var], dim=1)
            else:
                # (B, gene_size, output_dim) -> (B, output_dim)
                if self.gene_size == self.output_dim:
                    # translated
                    indices = torch.arange(self.output_dim, device=x_out.device)
                    x_out = x_out[:, indices, indices]
                else:
                    # translated
                    if x_out.shape[1] >= self.output_dim:
                        x_out = x_out[:, :self.output_dim, :].mean(dim=1)
                    else:
                        x_out = x_out.mean(dim=1)
        
        # translated (x_out, delta_s_hat) translated
        return x_out, delta_s_hat
    
    def convert_to_fp16(self):
        """translated FP16"""
        convert_module_to_f16(self)
    
    def convert_to_fp32(self):
        """translated FP32"""
        convert_module_to_f32(self)


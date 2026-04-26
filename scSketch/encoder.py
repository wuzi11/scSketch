import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .low_rank_drug_operator import LowRankDrugOperator


class EncoderMLPModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        num_classes=None,
        use_drug_structure=False,
        drug_dimension=1024,
        comb_num=1,
        output_size=60,
        dropout=0.1,
        use_fp16=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dtype = th.float16 if use_fp16 else th.float32
        self.drug_dimension = drug_dimension
        self.use_drug_structure = use_drug_structure

        if num_classes is None:
            l1 = 0
        else:
            l1 = hidden_sizes

        self.fc1 = nn.Linear(input_size + l1, hidden_sizes)
        self.bn1 = nn.BatchNorm1d(hidden_sizes)
        self.bn2 = nn.BatchNorm1d(hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.fc3 = nn.Linear(hidden_sizes, output_size)

        self.label_embed = nn.Linear(1, hidden_sizes)

        if use_drug_structure:
            self.drug_operator = LowRankDrugOperator(
                state_dim=hidden_sizes,
                drug_dim=drug_dimension,
                rank=8,
                use_mlp=True,
            )
            self.drug_proj = None
            print("EncoderMLPModel: Using Low-Rank Drug Operator (rank=8)")
        else:
            self.drug_operator = None
            self.drug_proj = None

    def forward(self, x_start, label=None, drug_dose=None, control_feature=None):
        if control_feature is not None:
            x_start = control_feature

        if label is not None:
            label_emb = self.label_embed(label)
            x_start = th.concat([x_start, label_emb], axis=1)

        h = x_start.type(self.dtype)
        h = F.relu(self.bn1(self.fc1(h)))
        h = F.relu(self.bn2(self.fc2(h)))

        if drug_dose is not None and self.use_drug_structure:
            if self.drug_operator is not None:
                h = self.drug_operator(h, drug_dose)
            elif self.drug_proj is not None:
                drug_emb = self.drug_proj(drug_dose)
                h = h + drug_emb

        z_sem = self.fc3(h)
        return z_sem

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMILPL(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        attention_dim=128,
        num_classes=3,
        dropout=0.25,
        attention_branches=1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.num_classes = num_classes
        self.attention_branches = attention_branches

        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.attention_tanh = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh()
        )
        self.attention_sigmoid = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(attention_dim, attention_branches)

        #self.classifier = nn.Linear(hidden_dim * attention_branches, num_classes)
        self.classifier = nn.Linear(hidden_dim * attention_branches + 1, num_classes)
    """
    def forward(self, x):
        H = self.feature_projection(x)  # [N, D]
        A = self.attention_weights(
            self.attention_tanh(H) * self.attention_sigmoid(H)
        )  # [N, heads]
        A = A.transpose(1, 0)  # [heads, N]
        A = torch.softmax(A, dim=1)  # normalize over tiles
        M = torch.matmul(A, H)  # [heads, D]
        M = M.view(-1)  # flatten to [heads * D]
        out = self.classifier(M)  # [C]
        return out, A

    """
    def forward(self, x):
        H = self.feature_projection(x)  # [N, D]

        A = self.attention_weights(
            self.attention_tanh(H) * self.attention_sigmoid(H)
        )  # [N, heads]
        A = A.transpose(1, 0)  # [heads, N]
        A = torch.softmax(A, dim=1)  # normalize over tiles

        M = torch.matmul(A, H)  # [heads, D]
        M = M.view(-1)  # [heads * D]

        # Add log(bag size) as an extra scalar feature
        bag_size = x.shape[0]
        log_bag_size = torch.log(torch.tensor(bag_size, dtype=torch.float32, device=x.device))
        M = torch.cat([M, log_bag_size.unsqueeze(0)], dim=0)  # shape: [heads * D + 1]

        out = self.classifier(M)  # [num_classes]
        return out, A
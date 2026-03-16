from torch import Tensor, nn, optim
import torch.nn.functional as F

class TextAdapter(nn.Module):
    """
    Its called Text Adapter, but we actually also have layers to adapt img embeddings. It was a quick patch.
    Naming needs to be cleaned
    """
    def __init__(self, dim: int, 
                 hidden_dim: int = 512, 
                 return_identity: bool =False, 
                 return_identity_im: bool = True 
                 ):

        super().__init__()
        self.return_identity = return_identity
        self.return_identity_im = return_identity_im

        if not self.return_identity:
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )

        if not self.return_identity_im:
            self.mlp_im = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )

    def forward(self, x, modality: str = "text"):
        
        if modality=="text":
            if self.return_identity:
                return F.normalize(x, dim=-1)
            x = x + self.mlp(x)
        elif modality=="image":
            if self.return_identity_im:
                return F.normalize(x, dim=-1)
            x = x + self.mlp_im(x)
        else:
            ValueError(f"Unsupported modality: {modality}")
        return F.normalize(x, dim=-1)
    
   
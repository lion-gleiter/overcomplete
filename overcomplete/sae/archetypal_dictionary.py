"""
Module dedicated to everything around the Dictionary Layer of SAE.
"""

import torch
from torch import nn
import torch.nn.functional as F


class RelaxedArchetypalDictionary(nn.Module):
    """
    Dictionary used for Relaxed Archetypal SAE (RA-SAE).

    Constructs a dictionary where each atom is a convex combination of data
    points from C, with a small relaxation term constrained by delta.

    For more details, see the paper:
    "Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction
    in Large Vision Models" by Fel et al. (2025), https://arxiv.org/abs/2502.12892

    Parameters
    ----------
    in_dimensions : int
        Dimensionality of the input data (e.g number of channels).
    nb_concepts : int
        Number of components/concepts in the dictionary. The dictionary is overcomplete if
        the number of concepts > in_dimensions.
    points : tensors
        Real data points (or point in the convex hull) used to find the candidates archetypes.
    delta : float, optional
        Constraint on the relaxation term, by default 1.0.
    use_multiplier : bool, optional
        Whether to train a positive scalar to multiply the dictionary after convex combination,
        making the dictionary in the conical hull (and not convex hull) of the data points,
        by default True.
    device : str, optional
        Device to run the model on ('cpu' or 'cuda'), by default 'cpu'.
    """

    def __init__(
        self, in_dimensions, nb_concepts,
        points, delta=1.0, use_multiplier=True,
        device="cpu",
        seed=0,
    ):
        super().__init__()
        self.in_dimensions = in_dimensions
        self.nb_concepts = nb_concepts
        self.device = device
        self.delta = delta

        # follow the eq.4 of the paper
        self.register_buffer("C", points.to(self.device), )
        self.nb_candidates = self.C.shape[0]
        self.W = nn.Parameter(torch.eye(nb_concepts, self.nb_candidates, device=device))
        self.sparse_W_init(points.to(self.device), 
                           n_atoms=nb_concepts, 
                           n_neighbors=points.shape[0] // nb_concepts + 1, 
                           temperature=0.1, 
                           seed=seed)
        self.Relax = nn.Parameter(torch.zeros(nb_concepts, self.in_dimensions, device=device))

        # init multiplier
        if use_multiplier:
            self.multiplier = nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)
        else:
            # by default, the multiplier will be exp(0) = 1 and not trainable
            self.register_buffer("multiplier", torch.tensor(0.0, device=device))

        self._fused_dictionary = None

    @torch.no_grad()
    def sparse_W_init(self, points, n_atoms, n_neighbors=8, temperature=0.1, seed=0):
        """Returns W of shape (n_atoms, n_candidates), row-stochastic and sparse."""
        C = F.normalize(points, dim=1)
        n_cand = C.shape[0]
        g = torch.Generator(device=C.device).manual_seed(seed)

        # Farthest-point sampling (cosine) -> diverse anchors
        anchors = torch.empty(n_atoms, dtype=torch.long, device=C.device)
        anchors[0] = torch.randint(n_cand, (1,), generator=g, device=C.device)[0]
        max_sim = C @ C[anchors[0]]
        for i in range(1, n_atoms):
            a = max_sim.argmin()
            anchors[i] = a
            max_sim = torch.maximum(max_sim, C @ C[a])

        # Each atom = softmax-weighted mix of its anchor and nearest neighbors
        W = torch.zeros(n_atoms, n_cand, device=C.device)
        for s in range(0, n_atoms, 1024):               # chunked to save memory
            sim = C[anchors[s:s + 1024]] @ C.T          # (chunk, n_cand)
            top_sim, top_idx = sim.topk(n_neighbors, dim=1)   # includes the anchor itself
            w = torch.softmax(top_sim / temperature, dim=1)
            W[s:s + 1024].scatter_(1, top_idx, w)

        self.W.data.copy_(W)
        # return W

    def forward(self, z):
        """
        Reconstruct input data from latent representation.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation tensor of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Reconstructed input tensor of shape (batch_size, dimensions).
        """
        dictionary = self.get_dictionary()
        x_hat = torch.matmul(z, dictionary)
        return x_hat

    @torch.no_grad()
    def project(self):
        with torch.autocast(device_type=str(self.device), dtype=torch.float16):
            self.W.clamp_(min=0)  # ReLU
            self.W.div_(self.W.sum(dim=-1, keepdim=True) + 1e-8)
            norm_Lambda = self.Relax.norm(dim=-1, keepdim=True)
            self.Relax.mul_(torch.where(norm_Lambda > 0, 
                                        (self.delta / norm_Lambda).clamp(max=1), 
                                        torch.zeros_like(norm_Lambda)))


    def get_dictionary(self):
        """
        Get the dictionary.

        Returns
        -------
        torch.Tensor
            The dictionary tensor of shape (nb_components, dimensions).
        """
        if self.training:
            # # we are in .train() mode, compute the dictionary on the fly
            # with torch.no_grad():
            #     # ensure W remains row-stochastic (positive and row sum to one)
            #     W = torch.relu(self.W)
            #     W /= (W.sum(dim=-1, keepdim=True) + 1e-8)
            #     self.W.data = W

            #     # enforce the norm constraint on Lambda to limit deviation from conv(C)
            #     norm_Lambda = self.Relax.norm(dim=-1, keepdim=True)  # norm per row
            #     scaling_factor = torch.where(norm_Lambda > 0.0, torch.clamp(self.delta / norm_Lambda, max=1), 0.0)  # safe scaling factor
            #     self.Relax.data *= scaling_factor  # scale Lambda to satisfy ||Lambda|| < delta

            # compute the dictionary as a convex combination plus relaxation
            # D = self.W @ self.C + self.Relax
            # return D * torch.exp(self.multiplier)
            with torch.autocast(device_type=str(self.device), dtype=torch.float16):
                return (self.W @ self.C + self.Relax) * torch.exp(self.multiplier)
        else:
            # we are in .eval() mode, return the fused dictionary
            assert self._fused_dictionary is not None, "Dictionary is not initialized."
            return self._fused_dictionary

    def train(self, mode=True):
        """
        Hook called when switching between training and evaluation mode.
        We use it to fuse W, C, Relax and multiplier into a single dictionary.

        Parameters
        ----------
        mode : bool, optional
            Whether to set the model to training mode or not, by default True.
        """
        if (not mode) and self.training:
            # we change to .eval() mode, fuse the dictionary
            self.project()
            with torch.no_grad():
                self._fused_dictionary = self.get_dictionary().detach()
        super().train(mode)

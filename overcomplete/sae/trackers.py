"""
Module dedicated to trackers for SAEs training.
"""

import torch


class DeadCodeTracker:
    """Track dead codes in an online manner."""

    def __init__(self, nb_concepts, device, *,
                 tracked_tokens=int(1e7)):
        """
        Initialize dead code tracker.

        Parameters
        ----------
        nb_concepts : int
            Number of features to track
        tracked_tokens : int
            A feature is dead if it has not been active in the last
            'tracked_tokens' tokens.
        device : str or torch.device
            Device to store the tracking tensor on
        """
        # self.alive_features = torch.zeros(nb_concepts, dtype=torch.bool, device=device)

        self.tracked_tokens = int(tracked_tokens)
        self.tokens_seen = 0

        # index of the last token on which each feature was active
        # (last_active==tracked_tokens means never active in the current token history).
        self.last_active = torch.ones(nb_concepts, dtype=torch.long, device=device) * self.tracked_tokens

    @property
    def warmed_up(self):
        """True once enough tokens were seen to fill the tracking horizon."""
        return self.tokens_seen >= self.tracked_tokens
    
    @torch.no_grad()
    def update(self, z):
        """
        Update alive features based on new activations.

        Parameters
        ----------
        z : torch.Tensor
            Encoded tensor of shape (batch_size, n_features)
        """
        # self.alive_features |= (z > 0).any(dim=0)
        n = z.shape[0]

        # 1-based position within the batch of the last token where each
        # feature fires (0 if it never fires in this batch).
        pos = torch.arange(n, device=z.device, dtype=torch.int32)
        last_pos = torch.where(z > 0, pos[:, None], self.tracked_tokens).amin(dim=0)

        self.last_active = torch.clamp(torch.minimum(self.last_active + n, last_pos), max=self.tracked_tokens)

        self.tokens_seen += n

    @property
    def alive_features(self):
        """Bool mask of features active within the last `tracked_tokens` tokens."""
        # Before warm-up, tokens_seen < tracked_tokens, so everything counts as alive.
        return self.last_active < self.tracked_tokens

    def get_dead_ratio(self):
        """
        Get the ratio of dead features.

        Returns
        -------
        float
            Ratio of dead features (between 0 and 1)
        """
        return 1 - self.alive_features.float().mean().item()

    def get_dead_indices(self):
        """Indices of currently dead features (e.g. for resampling)."""
        return torch.nonzero(~self.alive_features, as_tuple=True)[0]

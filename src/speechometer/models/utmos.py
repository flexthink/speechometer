"""UTMOS.
Authors
 * Jarod Duret 2024
 * Artem Ploujnikov 2024 (cosmetic changes only)
"""

from os import PathLike

import torch
import torch.nn as nn

from speechbrain.integrations.huggingface import Wav2Vec2

UTMOS_DEFAULT_JUDGE_ID = 288


class UTMOSModel(nn.Module):
    """The UTMOS model wrapper

    Arguments
    ---------
    source : str
        The WavLM source
    save_path : str | PathLike
        The path where the model will be saved
    features_dim : int, optional
        The features dimension
    num_domains : int, optional
        The number of domains
    domain_dim : int, optional
        The dimension of each domain
    num_judges : int, optional
        The number of "judges"
    judge_dim : int, optional
        The dimension of each judge
    decoder_hidden_size : int, optional
        The size of the decoder hidden state
    multiplier : float, optional
        The number that the raw model output is multiplied by
        to compute the score
    offset : float, optional
        The number that (raw output * multiplier) will be added
        to in order to get the score
    """

    def __init__(
        self,
        source: str,
        save_path: str | PathLike,
        features_dim: int = 768,
        num_domains: int = 3,
        domain_dim: int = 128,
        num_judges: int = 3000,
        judge_dim: int = 128,
        decoder_hidden_size: int = 512,
        multiplier: float = 2.0,
        offset: float = 3.0,
    ):
        super().__init__()

        self.ssl_encoder = Wav2Vec2(
            source,
            save_path,
            freeze=True,
            output_norm=False,
            freeze_feature_extractor=True,
            output_all_hiddens=False,
        )

        self.domain_embedding = nn.Embedding(num_domains, domain_dim)
        self.judge_embedding = nn.Embedding(num_judges, judge_dim)

        self.decoder = nn.LSTM(
            input_size=features_dim + domain_dim + judge_dim,
            hidden_size=decoder_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(decoder_hidden_size * 2, 2048),
            torch.nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1),
        )
        self.multiplier = multiplier
        self.offset = offset

    def forward(
        self,
        wav: torch.Tensor,
        domain_id: torch.Tensor | None = None,
        judge_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Computes the forward pass

        Arguments
        ---------
        wav : torch.Tensor
            The raw waveforms
        domain_id : torch.Tensor | None
            The domain identifiers
        judge_id : torch.Tensor | None
            The judge identifier

        Returns
        -------
        result : torch.Tensor
            The predicted rating(s)
        """

        if domain_id is None:
            domain_id = torch.zeros(
                len(wav), dtype=torch.int, device=wav.device
            )
        if judge_id is None:
            judge_id = (
                torch.ones(len(wav), dtype=torch.int, device=wav.device)
                * UTMOS_DEFAULT_JUDGE_ID
            )

        ssl_features = self.ssl_encoder(wav)
        domain_emb = self.domain_embedding(domain_id)
        judge_emb = self.judge_embedding(judge_id)

        domain_emb = domain_emb.unsqueeze(1).expand(
            -1, ssl_features.size(1), -1
        )
        judge_emb = judge_emb.unsqueeze(1).expand(-1, ssl_features.size(1), -1)
        concatenated_feature = torch.cat(
            [ssl_features, domain_emb, judge_emb], dim=2
        )

        decoder_output, _ = self.decoder(concatenated_feature)
        pred = self.classifier(decoder_output)

        return pred.mean(dim=1).squeeze(1) * self.multiplier + self.offset

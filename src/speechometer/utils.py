"""Utilities

Authors
 * Artem Ploujnikov 2026
"""

import torch


def undo_padding(
    batch: torch.Tensor, lengths: torch.Tensor
) -> list[torch.Tensor]:
    """Produces Python lists given a batch of sentences with
    their corresponding relative lengths.

    Adopted from SpeechBrain

    Modified not to undo tensors

    Arguments
    ---------
    batch : torch.Tensor
        Batch of sentences gathered in a batch.
    lengths : torch.Tensor
        Relative length of each sentence in the batch.

    Returns
    -------
    as_list : list[torch.Tensor]
        A python list of the corresponding input tensor.

    Example
    -------
    >>> batch = torch.rand([4, 100])
    >>> lengths = torch.tensor([0.5, 0.6, 0.7, 1.0])
    >>> snt_list = undo_padding(batch, lengths)
    >>> len(snt_list)
    4
    """
    batch_max_len = batch.shape[1]
    as_list = []
    for seq, seq_length in zip(batch, lengths):
        actual_size = int((seq_length * batch_max_len).round())
        seq_true = seq.narrow(0, 0, actual_size)
        as_list.append(seq_true)
    return as_list

from unittest.mock import MagicMock
from copy import deepcopy

import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence


class CopyingMock(MagicMock):
    def __call__(self, *args, **kwargs):
        args = deepcopy(args)
        kwargs = deepcopy(kwargs)
        return super().__call__(*args, **kwargs)


def populate_packed_sequence(batch_size: int = 3) -> PackedSequence:
    len_sequence = range(30, (30 - 2 * batch_size), -2)
    lengths = torch.tensor(len_sequence)
    sequences_vectors = []
    for length in lengths:
        sequences_vectors.append(torch.rand(length, 1))

    padded_sequences_vectors = pad_sequence(sequences_vectors)
    pack_padded_sequences_vectors = pack_padded_sequence(padded_sequences_vectors, lengths.cpu())
    return pack_padded_sequences_vectors

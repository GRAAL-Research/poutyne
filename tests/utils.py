"""
Copyright (c) 2022 Poutyne and all respective contributors.

Each contributor holds copyright over their respective contributions. The project versioning (Git)
records all such contribution source information.

This file is part of Poutyne.

Poutyne is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

Poutyne is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with Poutyne. If not, see
<https://www.gnu.org/licenses/>.
"""

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

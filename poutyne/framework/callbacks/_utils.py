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

import os
import warnings
from typing import Callable


def atomic_lambda_save(
    filename: str,
    save_lambda: Callable,
    args,
    *,
    temporary_filename: str = None,
    open_mode: str = 'w',
    atomic: bool = True,
):
    # pylint: disable=unspecified-encoding
    open_kwargs = dict(encoding='utf-8') if 'b' not in open_mode else {}
    if atomic:
        if temporary_filename is None:
            temporary_filename = filename + '.tmp'

        with open(temporary_filename, open_mode, **open_kwargs) as fd:
            save_lambda(fd, *args)

        try:
            os.replace(temporary_filename, filename)
        except OSError as e:
            # This may happen if the temp filesystem is not the same as the final destination's.
            warnings.warn(
                "Impossible to move the file to its final destination: "
                f"os.replace({temporary_filename}, {filename}) -> {e}"
            )
            os.remove(temporary_filename)

            warnings.warn(f'Saving {filename} non-atomically instead.')
            with open(filename, open_mode, **open_kwargs) as fd:
                save_lambda(fd, *args)
    else:
        with open(filename, open_mode, **open_kwargs) as fd:
            save_lambda(fd, *args)

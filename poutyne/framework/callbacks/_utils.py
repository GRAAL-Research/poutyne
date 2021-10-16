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

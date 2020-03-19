import os
import tempfile
import warnings
from typing import Callable


def atomic_lambda_save(filename: str,
                       save_lambda: Callable,
                       args,
                       *,
                       temporary_filename: str = None,
                       open_mode: str = 'w',
                       atomic: bool = True):
    if atomic:
        fd = None
        if temporary_filename is not None:
            fd = open(temporary_filename, open_mode)
            tmp_filename = temporary_filename
        else:
            fd = tempfile.NamedTemporaryFile(mode=open_mode, delete=False)
            tmp_filename = fd.name

        with fd:
            save_lambda(fd, *args)

        try:
            os.replace(tmp_filename, filename)
        except OSError as e:
            # This may happen if the temp filesystem is not the same as the final destination's.
            warnings.warn("Impossible to move the file to its final destination: "
                          "os.replace(%s, %s) -> %s" % (tmp_filename, filename, e))
            os.remove(tmp_filename)

            warnings.warn('Saving %s non-atomically instead.' % filename)
            with open(filename, open_mode) as fd:
                save_lambda(fd, *args)
    else:
        with open(filename, open_mode) as fd:
            save_lambda(fd, *args)

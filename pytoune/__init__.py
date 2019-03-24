import warnings
warnings.warn("PyToune has changed its name for Poutyne. Please use the new package name 'poutyne'. The 'pytoune' package will be removed in the next release.")

import sys
sys.modules[__name__] = __import__('poutyne')

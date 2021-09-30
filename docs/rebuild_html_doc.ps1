.  ../InvokeFunc.ps1
Invoke-NativeCommand sphinx-build -M clean source/ _build/ -W --keep-going
Invoke-NativeCommand sphinx-build -M html source/ _build/ -W --keep-going

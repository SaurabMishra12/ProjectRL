import sys
from .cli import main

try:
    raise SystemExit(main())
except (ValueError,PermissionError,RuntimeError,OSError,KeyError) as error:
    print(type(error).__name__+': '+str(error),file=sys.stderr)
    raise SystemExit(2)

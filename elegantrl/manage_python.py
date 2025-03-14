import sys
from typing import Tuple

# Define TupleAlias in a separate module
TupleAlias = tuple if sys.version_info >= (3, 9) else Tuple

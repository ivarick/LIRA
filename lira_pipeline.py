"""Public CLI and compatibility imports for the LIRA pipeline.

Implementation lives in the lira package.
"""

from lira.core import *  # noqa: F403
from lira.cases import *  # noqa: F403
from lira.statutes import *  # noqa: F403
from lira.selection import *  # noqa: F403
from lira.training import *  # noqa: F403
from lira.cli import *  # noqa: F403
from lira.cli import main


if __name__ == "__main__":
    main()

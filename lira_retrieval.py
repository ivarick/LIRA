"""Public CLI and compatibility imports for shared LIRA retrieval."""

from lira.retrieval_core import *  # noqa: F403
from lira.retrieval_candidates import *  # noqa: F403
from lira.retrieval_training import *  # noqa: F403
from lira.retrieval_cli import *  # noqa: F403
from lira.retrieval_cli import main

if __name__ == "__main__":
    main()

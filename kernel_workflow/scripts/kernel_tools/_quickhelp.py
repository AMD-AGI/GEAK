"""Progressive help for the shared kernel tools.

`-h` / `--help` prints a short card: why the tool is worth running, the two or three commands that
cover most uses, and how to read the output. `--help-full` prints the full argparse help (every flag
and the reasoning behind it). An agent that has never seen the tool should be able to run it from
the card alone; the full text is there when a flag or a refusal needs explaining.
"""
import sys


def handle(ap, card, argv=None):
    """Call before `ap.parse_args()`. Exits after printing when a help flag is present."""
    argv = sys.argv[1:] if argv is None else argv
    if "--help-full" in argv:
        ap.print_help()
        sys.exit(0)
    if "-h" in argv or "--help" in argv:
        print(card.strip("\n"))
        print("\n(--help-full: every flag, and why each check or refusal exists)")
        sys.exit(0)


def add_flag(ap):
    """Register --help-full so it shows in usage and is not rejected by the parser."""
    ap.add_argument("--help-full", action="store_true",
                    help="print the full help: every flag and the reasoning behind it")

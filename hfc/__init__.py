"""
hfc -- synthetic PEM fuel cell dataset generator and fault detector.

READ THIS FIRST. Every fault this package has ever seen is SYNTHETIC. There
is no labelled real fault data behind it, and no accuracy figure it reports
is verified on real faults. See docs/tool_design.md and MODEL_CARD.md.
"""

__version__ = "0.1.0"
GENERATOR_VERSION = "hfcgen-4.0"

UNVERIFIED = (
    "Real-fault accuracy is UNVERIFIED: every fault this tool was trained and "
    "evaluated on is synthetic. Treat its output as a screening aid, not a "
    "diagnosis.")

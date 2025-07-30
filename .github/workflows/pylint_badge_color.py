"""This module is called by the pylint.yml to determine the badge color
based on the pylint score."""

import os
import sys

score = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

if score >= 8.0:
    COLOR = "brightgreen"
elif score >= 6.0:
    COLOR = "yellow"
elif score >= 4.0:
    COLOR = "orange"
else:
    COLOR = "red"

with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as f:
    f.write(f"BADGE_COLOR={COLOR}\n")

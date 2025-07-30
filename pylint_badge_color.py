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

# Export to GitHub environment
env_path = os.environ.get("GITHUB_ENV")
if env_path:
    with open(env_path, "a", encoding="utf-8") as f:
        f.write(f"BADGE_COLOR={COLOR}\n")
    print(f"    ✅ BADGE_COLOR={COLOR} written to {env_path}")
else:
    print("   ⚠️ GITHUB_ENV is not set — failed to export BADGE_COLOR")

# # Expore to GitHub environment
# with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as f:
#     f.write(f"BADGE_COLOR={COLOR}\n")
#
# # Provide visual feedback in the GitHub Actions log
# # print(f"    Badge color set to '{COLOR}' for score {score}/10")
#
# # Provide visual feedback in the GitHub Actions log by actually querying
# # the GITHUB_ENV variable
# env_path = os.environ.get("GITHUB_ENV")
# if env_path:
#     with open(env_path, "r", encoding="utf-8") as f:
#         env_content = f.read()
#         if f"BADGE_COLOR={COLOR}" in env_content:
#             print(f"    ✅ Successfully exported BADGE_COLOR={COLOR} to GITHUB_ENV")
#         else:
#             print("    ⚠️ Failed to export BADGE_COLOR to GITHUB_ENV")
# else:
#     print("    ⚠️ GITHUB_ENV is not set — failed to export BADGE_COLOR")
#

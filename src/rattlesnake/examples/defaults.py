import os
import sys

# Define base directory
this_path = os.path.split(__file__)[0]
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    DIRECTORY = sys._MEIPASS  # pylint: disable=protected-access
else:
    DIRECTORY = this_path

NUM_CHANNELS = 21
NUM_FORCES = 9
SAMPLE_RATE = 2048
BUFFER_SIZE = 0.1
OUTPUT_OVERSAMPLE = 10
DAMPING_RATIO = 0.05

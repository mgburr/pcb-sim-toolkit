#!/usr/bin/env python3
"""Launch the PCB Simulation Toolkit GUI."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.gui.app import main

if __name__ == "__main__":
    main()

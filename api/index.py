import sys
import os

# Add parent directory to path to locate server.py and assets
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app

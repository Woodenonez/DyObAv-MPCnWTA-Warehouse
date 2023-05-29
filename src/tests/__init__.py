import os
import sys

# Add src/ to sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                       os.pardir))
sys.path.insert(0, src_dir)
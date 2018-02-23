import os

''' settings contains variables for directory roots and paths 
    so the project can be freely ported to different computers'''

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

print(ROOT_DIR)
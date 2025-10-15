"""
@Date         : 14-10-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core
@File         : logger.py
"""

import os

class Logger():
    def __init__(self):
        self.log_path = ""
    
    def setup(self, FILE_NAME, DEST_DIR):
        self.log_path = os.path.join(DEST_DIR, FILE_NAME + ".txt")
    
    def new_line(self, TEXT):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(TEXT + "\n")

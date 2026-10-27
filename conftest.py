import os

# Turn off OpenMDAO's automatic reports, which otherwise write a <problem name>_out directory for every test problem
os.environ.setdefault("OPENMDAO_REPORTS", "0")

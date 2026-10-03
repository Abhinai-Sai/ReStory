@echo off
py -3.13 create_basicsr_shim.py
py -3.13 fix_stylegan_shim.py
py -3.13 app.py

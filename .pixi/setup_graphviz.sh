#!/usr/bin/env sh

# The following error occurs without running 'dot -c' first:
#
# >>> import pydot as pd
# >>> pd.Dot.create(pd.Dot())
# "dot" with args ['-Tps', '/var/folders/cz/ysr8lt_90z37zyb25ds1sjcr0000gn/T/tmph_b89o11/tmpkcfldrb8'] returned code: 1
#
# stdout, stderr:
#  b''
# b'Format: "ps" not recognized. No formats found.\nPerhaps "dot -c" needs to be run (with installer\'s privileges) to register the plugins?\n'
#
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/Users/kratsg/pyhs3/.pixi/envs/py310/lib/python3.10/site-packages/pydot/core.py", line 1844, in create
#     assert process.returncode == 0, (
# AssertionError: "dot" with args ['-Tps', '/var/folders/cz/ysr8lt_90z37zyb25ds1sjcr0000gn/T/tmph_b89o11/tmpkcfldrb8'] returned code: 1

dot -c

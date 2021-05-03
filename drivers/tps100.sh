#!/bin/bash

for n in $(seq 1.0e9 1.0e9 5.5e9)
do
    python3 tpstest.py --mu "$n"
done
exit 0

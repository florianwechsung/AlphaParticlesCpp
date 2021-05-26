#!/bin/bash

for n in $(seq 1.0e9 0.5e9 3.5e9)
do
    python3 tpstest.py --mu "$n" --name "left" -s -d
done
exit 0

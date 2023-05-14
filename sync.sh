#!/bin/bash

# Uncomment to sync automatically
#( while inotifywait -e close_write main.py; do jupytext --to notebook main.py; done ) &
#( while inotifywait -e close_write main.ipynb; do jupytext --to py main.ipynb; done ) &
#wait

jupyter nbconvert --to script main.ipynb
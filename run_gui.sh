#!/bin/bash
# Launch script for Gomoku GUI with proper Python path

cd "$(dirname "$0")"
export PYTHONPATH="./python:$PYTHONPATH"
python python/gui/gomoku_gui.py
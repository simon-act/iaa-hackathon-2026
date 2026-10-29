#!/bin/bash
# Starts the autoresearch loop in the background, keeping the Mac awake (caffeinate).
#   bash autoresearch/run_loop.sh                 # Qwen model chosen automatically, 200 iterations
#   bash autoresearch/run_loop.sh --model qwen3.6 --iters 50
# To stop it:    touch autoresearch/STOP   (finishes the current experiment, then exits)
# To follow it:  tail -f autoresearch/loop.log     results: autoresearch/results.tsv
cd "$(dirname "$0")/.."
rm -f autoresearch/STOP
nohup caffeinate -dims .venv/bin/python -I autoresearch/run_loop.py "$@" >> autoresearch/loop.log 2>&1 &
echo "loop started (pid $!). Log: tail -f autoresearch/loop.log — stop: touch autoresearch/STOP"

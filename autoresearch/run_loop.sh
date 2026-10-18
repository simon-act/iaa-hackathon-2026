#!/bin/bash
# Avvia il loop autoresearch in background, con il Mac tenuto sveglio (caffeinate).
#   bash autoresearch/run_loop.sh                 # modello Qwen scelto in automatico, 200 iterazioni
#   bash autoresearch/run_loop.sh --model qwen3.6 --iters 50
# Per fermarlo:  touch autoresearch/STOP   (finisce l'esperimento in corso ed esce)
# Per seguirlo:  tail -f autoresearch/loop.log     risultati: autoresearch/results.tsv
cd "$(dirname "$0")/.."
rm -f autoresearch/STOP
nohup caffeinate -dims .venv/bin/python -I autoresearch/run_loop.py "$@" >> autoresearch/loop.log 2>&1 &
echo "loop avviato (pid $!). Log: tail -f autoresearch/loop.log — stop: touch autoresearch/STOP"

# Makefile for Part 1 - parallel matrix multiplication (OpenMP)
# `make` builds ./parallel. Needs gcc with OpenMP support (Linux / WSL).

CC      = gcc
CFLAGS  = -std=gnu11 -O3 -march=native -fopenmp -Wall -Wextra
LDLIBS  = -lm

.PHONY: all bench plot clean

all: parallel

parallel: parallel.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# Run every experiment (takes about 30-40 minutes), results go to ./results
bench: parallel
	python3 bench.py --out results

# Draw the graphs from ./results (needs python3-matplotlib)
plot:
	python3 plot.py --data results

clean:
	rm -f parallel
	rm -rf results __pycache__

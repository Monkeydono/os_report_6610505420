# Makefile for Part 2 - memory map visualizer
# mmap.py is plain Python 3 (standard library only) and needs no build step.
# `make` builds ./demo, a small C program whose memory map changes step by step.

CC     = gcc
CFLAGS = -std=gnu11 -O1 -g -Wall -Wextra -pthread

.PHONY: all run clean

all: demo

demo: demo.c
	$(CC) $(CFLAGS) -o $@ $<

# Real-time view of the demo program in the terminal
run: demo
	python3 mmap.py --run ./demo --watch 0.5

clean:
	rm -f demo *.html *.json

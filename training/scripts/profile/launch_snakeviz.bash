#!/usr/bin/env bash

# Takes a file path as an argument and runs snakeviz on it
# The extra arguments are there to make it work within Docker
snakeviz -H 0.0.0.0 -s -p $SNAKEVIZ_PORT $1

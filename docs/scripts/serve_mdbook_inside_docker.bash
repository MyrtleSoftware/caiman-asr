#!/usr/bin/env bash
mdbook-admonish install
mdbook serve --hostname 0.0.0.0 --port $1

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y cargo

RUN cargo install mdbook --version 0.4.37

RUN cargo install mdbook-admonish --version "1.15.0"

WORKDIR /code/docs

ENV PATH="${PATH}:/root/.cargo/bin"

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y curl git

WORKDIR /mdbook_packages

RUN curl -sSL \
    https://github.com/rust-lang/mdBook/releases/download/v0.4.40/mdbook-v0.4.40-x86_64-unknown-linux-gnu.tar.gz \
    | tar -xz --directory=/mdbook_packages

RUN curl -sSL \
    https://github.com/tommilligan/mdbook-admonish/releases/download/v1.18.0/mdbook-admonish-v1.18.0-x86_64-unknown-linux-gnu.tar.gz \
    | tar -xz --directory=/mdbook_packages

WORKDIR /code/docs

ENV PATH="${PATH}:/mdbook_packages"

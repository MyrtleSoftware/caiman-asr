# Documentation build

The documentation is built using [mdBook](https://rust-lang.github.io/mdBook/). They are built automatically and hosted with github pages.

## Local build

It is also possible to build the documentation locally. To do so:

1. Install mdBook:

```bash
cd docs
./scripts/install_mdbook.bash
```

2. Build and serve the docs:

```bash
./scripts/serve_mdbook.bash [optional port, default 3000]
```

The docs will now be available at `http://<machine_name>:3000` and will be re-built automatically when changes are made.

# CONTRIBUTING

This document contains information that is required to make contributions to the training repo.

## Developer install steps <a name="dev_install"></a>

We use pre-commit hooks to maintain PEP8 consistency across the repo.
Running pre-commit hooks requires having a python3 interpreter available in the same place that you commit - i.e. **outside the docker container**. To check that you have this first run the following in the same bash environment that use to run `git commit` commands:

```bash
which python3
which pip3
```

...and ensure that each returns a path. Python comes pre-installed on most Linux distributions so it's likely that even if you can't see it, it is on your system and can be added to your `$PATH`. If you don't have python3 installed you can install it on Ubuntu 18.04 or 20.04 by following [this tutorial](https://phoenixnap.com/kb/how-to-install-python-3-ubuntu). Now run:

```bash
pip install pre-commit==3.3.2
cd .. # navigate to the root of the repo
pre-commit install
```

Now, whenever you commit the hooks specified in `.pre-commit-config.yaml` will run on the files you have staged for commit.
The first time you run the hooks it may take a few minutes to download and install the code but subsequently, running the hooks shouldn't take more than a few seconds. If the pre-commits fail your commit will be aborted and you will have to commit again so it can be useful to commit using the following pattern:

```bash
git add <files> && git commit -m "message"
```

...so that you can re-run this command if the pre-commits fail without needing to re-add the files. It is possible to avoid running the hooks by running:

```bash
git commit -m "message" --no-verify
```

...but this is not recommended.

If you want to run the pre-commit manually you can do so with:

```bash
pre-commit run --all-files
```

## Running `pytest` tests <a name="pytest"></a>

If you would like to run the tests manually run `$ pytest` inside a running container.

## Type checking
We use [beartype](https://beartype.readthedocs.io/en/latest/) to check type annotations at runtime, and [jaxtyping](https://docs.kidger.site/jaxtyping/) to check tensor shapes.

When applying both decorators to one function, the order matters. The correct order is:
```python
@jaxtyped
@beartype
def function(...) -> ...:
   ...
```

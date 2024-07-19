# CONTRIBUTING

This document contains information that is required to make contributions to code in the [./training](./) directory. Before following the instructions below, please first follow the standard installation instructions: [\[docs\]](https://caiman-asr.myrtle.ai/training/installation.html) [\[markdown\]](./../docs/src/training/installation.md)

## Developer install steps <a name="dev_install"></a>

### Dependencies

We use pre-commit hooks to maintain PEP8 consistency across the repo.
Running pre-commit hooks requires having a python3 interpreter available in the same place that you commit - i.e. **outside the docker container**. To check that you have this first run the following in the same bash environment that use to run `git commit` commands:

```bash
which python3
which pip3
```

...and ensure that each returns a path. Python comes pre-installed on most Linux distributions so it's likely that even if you can't see it, it is on your system and can be added to your `$PATH`. If you don't have python3 installed you can install it on Ubuntu 18.04 or 20.04 by following [this tutorial](https://phoenixnap.com/kb/how-to-install-python-3-ubuntu).

Furthermore, please check that your Python version is 3.10 or higher by running:

```bash
python --version
```

### Installing pre-commit hooks

From the root of the repo run:

```bash
pip install pre-commit==3.3.2
pre-commit install
```

### Using pre-commit hooks

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

### Unit-tests

If you would like to run the tests manually run `$ pytest` inside a running container.

To check test coverage, run the following inside a container:

```
coverage run -m pytest tests
coverage report -m
```

### Distributed tests

There are tests to ensure that the batch splitting code is working correctly **in the distributed case** with number of GPUs > 1. The tests can be run on a multi-gpu machine from inside a running container with the following command:

```bash
python tests/rnnt/test_batch_split.py --num_gpus 2
```

After seeing the pass/fail pytest output you will need to KeyboardInterrupt this script otherwise it will run until the NCCL watchdog timer is up.

### Doctests

[Doctests](https://docs.python.org/3/library/doctest.html) can be run inside a container with `pytest caiman_asr_train`.

## Type checking

We use [beartype](https://beartype.readthedocs.io/en/latest/) to check type annotations at runtime, and [jaxtyping](https://docs.kidger.site/jaxtyping/) to check tensor shapes.

When applying jaxtyped, use it like this:

```python
@jaxtyped(typechecker=beartype)
def function(...) -> ...:
   ...
```

## Merging PRs

We typically squash and merge PRs instead of just merging them. Once your branch is merged, we recommend deleting it from GitHub.

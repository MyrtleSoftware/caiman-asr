from pathlib import Path


def persistent_cache(dir: str | Path) -> Path:
    """
    Preface `dir` with the persistent cache directory.

    In launch.sh `~/.cache/myrtle` is mounted outside the container.
    """
    return Path("~/.cache/myrtle/").expanduser() / Path(dir)

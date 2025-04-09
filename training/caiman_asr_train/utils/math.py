from beartype import beartype


@beartype
def ceil_div(x: int, *, by: int) -> int:
    """
    Pythons operator // implements floor division, i,e., the result is
    rounded towards negative infinity. This function implements ceil division,
    i.e., the result is rounded towards positive infinity.
    """
    return (x + by - 1) // by


@beartype
def round_down(x: int, *, multiple_of: int) -> int:
    """
    Round towards negative infinity to the nearest multiple of `multiple_of`.
    """
    if multiple_of <= 0:
        raise ValueError("Cannot round down a negative number")

    return (x // multiple_of) * multiple_of


@beartype
def round_up(x: int, *, multiple_of: int) -> int:
    """
    Round towards positive infinity to the nearest multiple of `multiple_of`.
    """
    if multiple_of <= 0:
        raise ValueError("Cannot round up a negative number")

    return ceil_div(x, by=multiple_of) * multiple_of

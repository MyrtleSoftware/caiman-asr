#!/usr/bin/env python3


import string

import pytest


@pytest.fixture()
def default_charset():
    charstr = " '" + string.ascii_lowercase
    return list(charstr)

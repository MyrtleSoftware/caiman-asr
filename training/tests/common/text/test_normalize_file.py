#!/usr/bin/env python3
import pytest

from rnnt_train.common.text.normalize_file import normalize, normalize_by_line


@pytest.mark.parametrize("n", [normalize, normalize_by_line])
def test_wishlist(n):
    """It would be nice if normalize handled these cases better. But
    also changing normalize will cause the LM datasets' caches to need
    rebuilding"""
    # TODO This one loses the dollar sign
    assert (
        n("$ 124,758") == "one hundred twenty four thousand seven hundred fifty eight"
    )
    # TODO This one is wrong
    assert n("$1.2") == "one dollar two cents"
    # TODO Newlines sometimes cause issues because they're removed after
    # numbers are parsed
    assert n("$6.2\n million") == "six dollars two cents million"
    assert n("$6.2\nmillion") == "six dollars two cents million"


@pytest.mark.parametrize("n", [normalize, normalize_by_line])
def test_both(n):
    # Dollars and percents are usually handled well:
    assert (
        n("$124,758")
        == "one hundred twenty four thousand seven hundred fifty eight dollars"
    )
    assert n("$6.2 million") == "six point two million dollars"
    assert n("$1.2 million") == "one point two million dollars"
    assert (
        n("34.2%, to $19,111,001 and 12%")
        == "thirty four point two percent to nineteen million one hundred eleven thousand one dollars and twelve percent"
    )
    assert (
        n(" approximately $1.20 per share. ")
        == "approximately one dollar twenty cents per share"
    )
    assert (
        n("I earned $101 in 1999!")
        == "i earned one hundred one dollars in nineteen ninety nine"
    )
    assert n("$6 million") == "six million dollars"
    assert n("4.5 million dollars") == "four point five million dollars"
    assert n("$4.5 million ") == "four point five million dollars"
    assert n("$5.0 million ") == "five point zero million dollars"
    assert n("$4.5 billion ") == "four point five billion dollars"
    assert n("$4.5 trillion ") == "four point five trillion dollars"
    assert n("$4.5 thousand ") == "four point five thousand dollars"
    assert n("$450 thousand ") == "four hundred fifty thousand dollars"
    # Time isn't handled well, but it's acceptable:
    assert n("3:05 right now") == "three zero five right now"
    # Dashes are deleted if alone, but replaced with space if joining words:
    dash1 = "-"
    dash2 = "—"
    dash3 = "–"
    assert dash1 != dash2
    assert dash1 != dash3
    assert dash2 != dash3
    assert n(dash1) == ""
    assert n(dash2) == ""
    assert n(dash3) == ""
    rnnt1 = "RNN-T"
    rnnt2 = "RNN—T"
    rnnt3 = "RNN–T"
    assert rnnt1 != rnnt2
    assert rnnt1 != rnnt3
    assert rnnt2 != rnnt3
    assert n(rnnt1) == "rnn t"
    assert n(rnnt2) == "rnn t"
    assert n(rnnt3) == "rnn t"
    # Most special characters are deleted, except @ % & + are expanded
    assert n("! @ # $ % ^ & * ( ) +") == "at percent and plus"
    assert n("example@website.com") == "example at website com"
    # Non-Latin characters are transliterated character-by-character, which is imperfect:
    assert n("你好 means hello") == "ni hao means hello"
    assert n("こんにちは also means hello") == "konnitiha also means hello"
    assert n("être is a french verb") == "etre is a french verb"
    assert n("epsilon is ε") == "epsilon is e"
    # Typically Чебышёв is Anglicized as Chebyshev
    assert n("Пафнутий Львович Чебышёв") == "pafnutii l'vovich chebyshiov"
    # Unnecessary spaces are removed
    assert n("   hello   there    ") == "hello there"
    # New lines become spaces
    assert n("new\nline") == "new line"
    assert n("1\n2\n3") == "one two three"


def test_normalize_by_line():
    # A single "\n" becomes " "
    assert normalize_by_line("\n") == " "
    # Unparseable lines are skipped:
    assert (
        normalize_by_line(
            "$11.00 is parseable\n$12.,00is not parseable\n$13.01 is parseable",
            quiet=True,
        )
        == "eleven dollars is parseable  thirteen dollars one cent is parseable"
    )


def test_normalize():
    # A single "\n" becomes ""
    assert normalize("\n") == ""

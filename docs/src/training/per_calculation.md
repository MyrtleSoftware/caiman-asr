# PER Calculation

#### PER Formula

The Punctuation Error Rate (PER) is calculated with the same formula as the WER.
It is the count of additions, deletions,
and substitutions of punctuation marks divided by the number of punctuation marks in the
reference text.

Currently, the default punctuation marks used are: `,` `.` `?`.
They can be amended by parsing a different list with the `--punctuation_marks` argument.

The punctuation normalisation rules are the following:

```
    `;` -> `,` semi-colon to comma
    `:` -> `.` colon to period
    `!` -> `.` exclamation mark to period
    `-` -> ` ` hyphen to space
```

This means that both hypotheses and references are normalized before the calculation.
It can be disabled by parsing the `--punctuation_normalization=False` argument.

For example:

```
Hypothesis: Hi  dear. Nice to see you, how are you
Reference:  Hi, dear! Nice to see you. How are you?
```

The punctuation error rate is calculated as:
\$$
PER = \\frac{S + D + I}{N} \\times 100 = \\frac{1 + 2 + 0}{4} \\times 100=75\\%
$\$

where `S` is the number of substitutions, `D` is the number of deletions,
`I` is the number of insertions, and `N=4` is the number of punctuation marks in the reference text.

Note that in the above example the substitution of `.` for `!` is not counted
due to the punctuation normalization.

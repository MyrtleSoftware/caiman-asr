# Keyword boosting

Keyword boosting is a technique to improve the recognition of domain-specific
words/phrases and proper nouns. It works by boosting/suppressing the
probability of specific tokens at inference time.

Keyword boosting is currently only available in the beam decoder. To use
keyword boosting create a `json` file containing your keywords and their
corresponding boost values. The `json` file should have the following format:

```json
{
  "keywords": {
    "keyword": <exponential boost factor>,
  }
}
```

Keywords are case and space sensitive, they should be formatted using the same
character-set as the output of your decoder. The boost factors should be
numeric values. The typical boost factors are in the range -1 to 1.

As an example to discuss how keyword boosting works, consider the following
example (__note:__ the empty spaces are important as the keywords are space
sensitive):

```json
{
  "keywords": {
    "car": 1.0,
    " cat": 2.0,
    " bat ": -1.0,
  }
}
```

This would increase the probability of words containing the sequence `car`;
increase the probability of words starting with `cat` (more strongly than the
`car` boost); and decrease the probability of the __whole__ word `bat`.

Picking the boost values is a domain-specific task that requires some trial and
error.

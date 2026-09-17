#!/usr/bin/env python3
import string

import pytest

from caiman_asr_train.data.text.normalizers import lowercase_normalize, standardize_text
from caiman_asr_train.data.text.whisper_text_normalizer import EnglishTextNormalizer


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (("hello, lemme help"), ("hello let me help")),
        (("wanna see today's menu"), ("want to see today's menu")),
        (("I'd have what you'd have"), ("i would have what you would have")),
        (("utilize will be utilise"), ("utilize will be utilize")),
        (("Aren't <Silence> [inaudible] (audible)"), ("are not audible")),
        (("£ 20"), ("20")),
        (
            ("hmm that's what we'll normalise in today's example"),
            ("that is what we will normalize in today's example"),
        ),
        (("mmhmm uh-huh cuz"), ("because")),
        (("hmmm ummmmmmmmmm"), ("")),
        (("uhhh uhhuhh yes hmmmm huh huhh"), ("yes")),
        (("mmmhm"), ("")),
        (("Hello, I'm a human"), ("hello i am a human")),
        (
            ("won't can't let's lemme dunno"),
            ("will not can not let us let me do not know"),
        ),
        (("woulda coulda shoulda"), ("would have could have should have")),
        (("i'ma and imma"), ("i am going to and i am going to")),
        (
            ("mr Smith, pres Obama, dr. Jones, mrs Smith"),  # codespell:ignore
            ("mister smith president obama doctor jones missus smith"),
        ),
        (
            ("he'd been she's been gen. Walker's got"),
            ("he had been she has been general walker has got"),
        ),
        (("hasn't they're it's he's"), ("has not they are it is he is")),
        (
            ("That's what's going on and there's a lot of stuff."),
            ("that is what is going on and there is a lot of stuff"),
        ),
        (("they'd she'll i'm I've"), ("they would she will i am i have")),
        (("immaterial imm i-mm i'm i'mm"), ("immaterial imm i i am i'")),
        (
            ("We represent individual, um. The fastest way is meetings"),
            ("we represent individual the fastest way is meetings"),
        ),
        (
            ('"We represent individual, um. The fastest way is meetings"'),
            ("we represent individual the fastest way is meetings"),
        ),
        (("screen 's"), ("screen's")),
        (("INTERNALLY 'CAUSE"), ("internally because")),
        (("This 'is not' a contraction"), ("this 'is not' a contraction")),
    ],
)
def test_whisper(test_input, expected):
    normalizer = EnglishTextNormalizer()
    returned = normalizer(test_input)
    assert expected == returned


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (("hello, lemme help"), ("hello, let me help")),
        (("wanna see today's menu"), ("want to see today's menu")),
        (("I'd have what you'd have"), ("I would have what you would have")),
        (("utilize will be utilise"), ("utilize will be utilize")),
        (("Aren't <Silence> [inaudible] (audible)"), ("Are not (audible)")),
        (("£ 20"), ("£ 20")),
        (
            ("hmm that's what we'll normalise in today's example"),
            ("that is what we will normalize in today's example"),
        ),
        (("mmhmm uh-huh cuz"), ("because")),
        (("hmmm ummmmmmmmmm"), ("")),
        (("uhhh uhhuhh yes hmmmm huh huhh"), ("yes")),
        (("mmmhm"), ("")),
        (("Hello, I'm a human"), ("Hello, I am a human")),
        (
            ("won't can't let's lemme dunno"),
            ("will not can not let us let me do not know"),
        ),
        (("woulda coulda shoulda"), ("would have could have should have")),
        (("i'ma and imma"), ("i am going to and i am going to")),
        (
            ("Mr Smith, Pres Obama, dr. Jones, mrs Smith"),  # codespell:ignore
            ("Mister Smith, President Obama, doctor Jones, missus Smith"),
        ),
        (
            ("he'd been she's been gen. Walker's got"),
            ("he had been she has been general Walker has got"),
        ),
        (("hasn't they're it's he's"), ("has not they are it is he is")),
        (
            ("That's what's going on and there's a lot of stuff."),
            ("That is what is going on and there is a lot of stuff."),
        ),
        (("they'd she'll i'm I've"), ("they would she will i am I have")),
        (("immaterial imm i-mm i'm i'mm"), ("immaterial imm i i am i'")),
        (
            ("ald. jones and ald Smith joined Lt. col Hanson."),
            ("alderman jones and alderman Smith joined Lieutenant colonel Hanson."),
        ),
        (("IMMA BE A HUMAN!!"), ("I AM GOING TO BE A HUMAN!!")),
        (("MR. SMITH IT'S TIME TO GET UP!!"), ("MISTER SMITH IT IS TIME TO GET UP!!")),
        (("Mr. Smith it's time to get up."), ("Mister Smith it is time to get up.")),
        (("Hmm hmmm mm UMM m"), ("m")),
        (("Yes. Hmmm, mm."), ("Yes.")),
        (
            ("We represent individual, um. The fastest way is meetings"),
            ("We represent individual. The fastest way is meetings"),
        ),
        (
            ('"We represent individual, um. The fastest way is meetings"'),
            ('"We represent individual. The fastest way is meetings"'),
        ),
        (
            (
                "We represent individual. um. The fastest way is meetings! HMM. Okay, umm."
            ),
            ("We represent individual. The fastest way is meetings! Okay."),
        ),
        (("crummm Umm. !HMM. HMMEEEK, hi..Umm-mmmm"), ("crummm. HMMEEEK, hi.")),
        (
            ("We represent individual... um. things."),
            ("We represent individual... things."),
        ),
        (
            (
                "We represent individual. um. The fastest way is meetings! HMM. umm. mmhmm?"
            ),
            ("We represent individual. The fastest way is meetings!"),
        ),
        (("HELLO??!!"), ("HELLO?")),
        (("HELLO??"), ("HELLO??")),
        (("HELLO..."), ("HELLO...")),
        (("HELLO...!!??"), ("HELLO.")),
        (("No. Um, um, well"), ("No. Well")),
        (
            ("hmmm? Capitalise the next word. ummm.. hi"),
            ("Capitalize the next word. Hi"),
        ),
        (("UH MM-HMM UH HUH-HUH MMMM. HMMM?!"), ("")),
        (("HMM."), ("")),
        (("hi umm."), ("hi.")),
        (("Hi. umm. I am a dr. Hmm. Ummm! Mmmhmm."), ("Hi. I am a doctor.")),
        (
            (
                "Dr. umm Smith mmhmm umm, yes please, MMM. "
                "Prof. um Smith and umm prof. umm, yes please."
            ),
            ("Doctor Smith yes please. Professor Smith and professor yes please."),
        ),
        (("SCREEN 'S"), ("SCREEN'S")),
        (("INTERNALLY 'CAUSE"), ("INTERNALLY BECAUSE")),
        ("COVID-19", "COVID-19"),
        (("'Cause it seems"), ("Because it seems")),
        (("I, SHOULDN'T"), ("I, SHOULD NOT")),
        (("I do. do"), ("I do. Do")),
        (("I U.S.A. do"), ("I U.S.A. do")),
    ],
)
def test_whisper_punc_cased(test_input, expected):
    normalizer = EnglishTextNormalizer(preserve_case=True, preserve_punctuation=True)
    returned = normalizer(test_input)
    assert expected == returned


@pytest.mark.parametrize(
    "test_input, expected1, expected2",
    [
        (("hello, lemme help"), ("hello, let me help"), ("hello, lemme help")),
        (
            ("wanna see today's menu"),
            ("want to see today's menu"),
            ("wanna see today's menu"),
        ),
        (
            ("I'd have what you'd have"),
            ("I would have what you would have"),
            ("I'd have what you'd have"),
        ),
        (
            ("that's how I'd utilise it cuz"),
            ("that is how I would utilize it because"),
            ("that's how I'd utilize it cuz"),
        ),
        (
            ("prof that's wrong!"),
            ("professor that is wrong!"),
            ("professor that's wrong!"),
        ),
    ],
)
def test_whisper_expand_contractions(test_input, expected1, expected2):
    normalizer = EnglishTextNormalizer(
        preserve_case=True, preserve_punctuation=True, expand_contractions=True
    )
    returned = normalizer(test_input)
    assert expected1 == returned
    normalizer = EnglishTextNormalizer(
        preserve_case=True, preserve_punctuation=True, expand_contractions=False
    )
    returned = normalizer(test_input)
    assert expected2 == returned


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("hello, lemme help", "hello let me help"),
        ("wanna see today's menu", "want to see today's menu"),
        ("I'd have what you'd have", "i would have what you would have"),
        ("utilize will be utilise", "utilize will be utilize"),
        (
            "Aren't <Silence> [audible] (audible)",
            "are not audible audible",
        ),
        ("£ 20", "ps twenty"),
        ("$ 20", "twenty"),
        ("$20", "twenty dollars"),
        (
            "hmm that's what we'll standardise in today's example",
            "that is what we will standardize in today's example",
        ),
        ("Dr. Smith", "doctor smith"),
        ("$1.02", "one dollar two cents"),
        ("café", "cafe"),
        ("cats & dogs", "cats and dogs"),
        ("<tags> <will_disappear>", ""),
        ("COVID-19", "covid nineteen"),
        ("That's -1 for you", "that is minus one for you"),
        ("-1 for you", "minus one for you"),
        ("2-1 for you", "two to one for you"),
        ("20%", "twenty percent"),
        ("20 %", "twenty percent"),
    ],
)
def test_normalize_standardize(test_input, expected):
    charset = list(" '<>" + string.ascii_lowercase)
    test_input = lowercase_normalize(test_input, charset=charset, lang_code="en")
    returned = standardize_text((test_input), lang_code="en")
    assert expected == returned

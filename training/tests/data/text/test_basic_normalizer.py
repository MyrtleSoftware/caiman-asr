#!/usr/bin/env python3
import pytest

from caiman_asr_train.data.text.allowed_charset import get_allowed_charset
from caiman_asr_train.data.text.basic_normalizer import BasicTextNormalizer
from caiman_asr_train.data.text.normalizers import lowercase_normalize, standardize_text


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (("Aren't <Silence> [inaudible] (audible)"), ("aren't")),
        (("£ 20"), ("£ 20")),
        (
            ("FR: Il rompit le premier ce silence. — Vous me disiez donc de revenir ?"),
            ("fr il rompit le premier ce silence vous me disiez donc de revenir"),
        ),
        (
            ("FR: Il grommelait: fille d'or, épouseur de cuivre."),  # codespell:ignore
            ("fr il grommelait fille d'or épouseur de cuivre"),  # codespell:ignore
        ),
        (("Test if symbol gets converted: 𝖧"), ("test if symbol gets converted h")),
        (
            ("EL: τον Ιανουάριο του 2012 μεταξύ Ευρωπαϊκή Ένωση και Κοσμοπεδίου."),
            ("el τον ιανουάριο του 2012 μεταξύ ευρωπαϊκή ένωση και κοσμοπεδίου"),
        ),
        (
            (
                "BG: Благодаря и на колегите за конструктивните изказвания, "
                "за многоценностните коментари."
            ),
            (
                "bg благодаря и на колегите за конструктивните изказвания "
                "за многоценностните коментари"
            ),
        ),
        (
            (
                "HR: Da nema Kažnjavanja nasilja nad ženama, "
                "mislim primjerice na promjenu članka 306."
            ),
            (
                "hr da nema kažnjavanja nasilja nad ženama "
                "mislim primjerice na promjenu članka 306."
            ),
        ),
        (
            ("SL: Vidite, kaj lahko najdete v evropskih časopisih."),
            ("sl vidite kaj lahko najdete v evropskih časopisih"),
        ),
        (
            ("SK: Myslím, že odpoveď bola jednoduchá. Ja neviem, či ste to pochopili."),
            ("sk myslím že odpoveď bola jednoduchá ja neviem či ste to pochopili"),
        ),
        (("EN: i'ma and imma"), ("en i'ma and imma")),
        (
            (
                "HU: Végül a 19-es módosítóval kapcsolatban elmondanám, "
                "hogy itt egy nemzetközi megállapodást sürget."
            ),
            (
                "hu végül a 19-es módosítóval kapcsolatban elmondanám "
                "hogy itt egy nemzetközi megállapodást sürget"
            ),
        ),
        (
            ("PT: 27 e ENVI 50, idênticas. Ah, não."),
            ("pt 27 e envi 50 idênticas ah não"),
        ),
        (
            ("ET: Euroopa Liidu võimekus nõuda reforme läbi viia on piiratud."),
            ("et euroopa liidu võimekus nõuda reforme läbi viia on piiratud"),
        ),
        (
            ("NL: 48% van de Europeanen was het daarmee eens."),
            ("nl 48% van de europeanen was het daarmee eens"),
        ),
        (
            ("PL: Uwzględnimy również przyszłe debaty, wcześniejsze debaty."),
            ("pl uwzględnimy również przyszłe debaty wcześniejsze debaty"),
        ),
        (
            ("IT: La parola adesso al coordinatore dell'Econ, Winberg."),
            ("it la parola adesso al coordinatore dell'econ winberg"),
        ),
        (
            (
                "DE: Jetzt weiß ich nicht, was Herr Lewandowski erklärt "
                "hat zum GAP-Haushalt, dass dessen Anteil am "
                "Gesamthaushalt zurückgehen muss."
            ),
            (
                "de jetzt weiß ich nicht was herr lewandowski erklärt "
                "hat zum gap-haushalt dass dessen anteil am "
                "gesamthaushalt zurückgehen muss"
            ),
        ),
        (
            ("LV: ārjo politikas pētniecības budžetas strādāt."),
            ("lv ārjo politikas pētniecības budžetas strādāt"),
        ),
        (
            (
                "MT: Ejjew, ma t-oħduġ t-istahbi għara s-sobajkom "
                "u tajdu li ma jimx alternativi."
            ),
            (
                "mt ejjew ma t-oħduġ t-istahbi għara s-sobajkom "
                "u tajdu li ma jimx alternativi"
            ),
        ),
        (
            (
                "DA: Så altså det bliver ikke billigere, "
                "men det kan helt sikkert blive mere."
            ),
            (
                "da så altså det bliver ikke billigere "
                "men det kan helt sikkert blive mere"
            ),
        ),
        (
            ("FI: Tarkistus 51 puolesta vastaan tyhjää, hyväksytty."),
            ("fi tarkistus 51 puolesta vastaan tyhjää hyväksytty"),
        ),
        (
            ("LT: Ką mes norėtume matyti gyventint ant stabilumo priemonę? "),
            ("lt ką mes norėtume matyti gyventint ant stabilumo priemonę"),
        ),
        (
            (
                "CS: V určité chvíli to musí převzít žalobci, "
                "jakmile dojde ke spáchání trestného činu."
            ),
            (
                "cs v určité chvíli to musí převzít žalobci "
                "jakmile dojde ke spáchání trestného činu"
            ),
        ),
        (
            (
                "SV: Nästa vecka drar vi igång Rapid Alert-systemet "
                "som finns med i handlingsplanerna också."  # codespell:ignore
            ),
            (
                "sv nästa vecka drar vi igång rapid alert-systemet "
                "som finns med i handlingsplanerna också"  # codespell:ignore
            ),
        ),
        (
            (
                "RO: Nu uitați să dați like, să lăsați un comentariu și să "
                "distribuiți acest material video pe alte rețele sociale."
            ),
            (
                "ro nu uitați să dați like să lăsați un comentariu și să "
                "distribuiți acest material video pe alte rețele sociale"
            ),
        ),
        (
            (
                "ES: ¿Tiene previsto que este proyecto piloto se "
                "extienda a otros Estados miembros?"
            ),
            (
                "es tiene previsto que este proyecto piloto se "
                "extienda a otros estados miembros"
            ),
        ),
        (
            (
                "ES: ¿Tiene previsto «test» que este proyecto piloto se "
                "extienda a otros Estados miembros?"
            ),
            (
                "es tiene previsto test que este proyecto piloto se "
                "extienda a otros estados miembros"
            ),
        ),
    ],
)
def test_basic(test_input, expected):
    normalizer = BasicTextNormalizer()
    returned = normalizer(test_input)
    assert expected == returned


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (("Aren't <Silence> [inaudible] (audible)"), ("Aren't")),
        (("£ 20"), ("£ 20")),
        (
            ("FR: Il rompit le premier ce silence. — Vous me disiez donc de revenir ?"),
            ("FR: Il rompit le premier ce silence. — Vous me disiez donc de revenir ?"),
        ),
        (
            ("FR: Il grommelait: fille d'or, épouseur de cuivre."),  # codespell:ignore
            ("FR: Il grommelait: fille d'or, épouseur de cuivre."),  # codespell:ignore
        ),
        (
            ("Test that symbol isn't converted: 𝖧"),
            ("Test that symbol isn't converted: 𝖧"),
        ),
        (
            ("EL: τον Ιανουάριο του 2012 μεταξύ Ευρωπαϊκή Ένωση και Κοσμοπεδίου."),
            ("EL: τον Ιανουάριο του 2012 μεταξύ Ευρωπαϊκή Ένωση και Κοσμοπεδίου."),
        ),
        (
            (
                "BG: Благодаря и на колегите за конструктивните изказвания, "
                "за многоценностните коментари."
            ),
            (
                "BG: Благодаря и на колегите за конструктивните изказвания, "
                "за многоценностните коментари."
            ),
        ),
        (
            (
                "HR: Da nema Kažnjavanja nasilja nad ženama, "
                "mislim primjerice na promjenu članka 306."
            ),
            (
                "HR: Da nema Kažnjavanja nasilja nad ženama, "
                "mislim primjerice na promjenu članka 306."
            ),
        ),
        (
            ("SL: Vidite, kaj lahko najdete v evropskih časopisih."),
            ("SL: Vidite, kaj lahko najdete v evropskih časopisih."),
        ),
        (
            ("SK: Myslím, že odpoveď bola jednoduchá. Ja neviem, či ste to pochopili."),
            ("SK: Myslím, že odpoveď bola jednoduchá. Ja neviem, či ste to pochopili."),
        ),
        (("EN: i'ma and imma"), ("EN: i'ma and imma")),
        (
            (
                "HU: Végül a 19-es módosítóval kapcsolatban elmondanám, "
                "hogy itt egy nemzetközi megállapodást sürget."
            ),
            (
                "HU: Végül a 19-es módosítóval kapcsolatban elmondanám, "
                "hogy itt egy nemzetközi megállapodást sürget."
            ),
        ),
        (
            ("PT: 27 e ENVI 50, idênticas. Ah, não."),
            ("PT: 27 e ENVI 50, idênticas. Ah, não."),
        ),
        (
            ("ET: Euroopa Liidu võimekus nõuda reforme läbi viia on piiratud."),
            ("ET: Euroopa Liidu võimekus nõuda reforme läbi viia on piiratud."),
        ),
        (
            ("NL: 48% van de Europeanen was het daarmee eens."),
            ("NL: 48% van de Europeanen was het daarmee eens."),
        ),
        (
            ("PL: Uwzględnimy również przyszłe debaty, wcześniejsze debaty."),
            ("PL: Uwzględnimy również przyszłe debaty, wcześniejsze debaty."),
        ),
        (
            ("IT: La parola adesso al coordinatore dell'Econ, Winberg."),
            ("IT: La parola adesso al coordinatore dell'Econ, Winberg."),
        ),
        (
            (
                "DE: Jetzt weiß ich nicht, was Herr Lewandowski erklärt "
                "hat zum GAP-Haushalt, dass dessen Anteil am "
                "Gesamthaushalt zurückgehen muss."
            ),
            (
                "DE: Jetzt weiß ich nicht, was Herr Lewandowski erklärt "
                "hat zum GAP-Haushalt, dass dessen Anteil am "
                "Gesamthaushalt zurückgehen muss."
            ),
        ),
        (
            ("LV: ārjo politikas pētniecības budžetas strādāt."),
            ("LV: ārjo politikas pētniecības budžetas strādāt."),
        ),
        (
            (
                "MT: Ejjew, ma t-oħduġ t-istahbi għara s-sobajkom "
                "u tajdu li ma jimx alternativi."
            ),
            (
                "MT: Ejjew, ma t-oħduġ t-istahbi għara s-sobajkom "
                "u tajdu li ma jimx alternativi."
            ),
        ),
        (
            (
                "DA: Så altså det bliver ikke billigere, "
                "men det kan helt sikkert blive mere."
            ),
            (
                "DA: Så altså det bliver ikke billigere, "
                "men det kan helt sikkert blive mere."
            ),
        ),
        (
            ("FI: Tarkistus 51 puolesta vastaan tyhjää, hyväksytty."),
            ("FI: Tarkistus 51 puolesta vastaan tyhjää, hyväksytty."),
        ),
        (
            ("LT: Ką mes norėtume matyti gyventint ant stabilumo priemonę?"),
            ("LT: Ką mes norėtume matyti gyventint ant stabilumo priemonę?"),
        ),
        (
            (
                "CS: V určité chvíli to musí převzít žalobci, "
                "jakmile dojde ke spáchání trestného činu."
            ),
            (
                "CS: V určité chvíli to musí převzít žalobci, "
                "jakmile dojde ke spáchání trestného činu."
            ),
        ),
        (
            (
                "SV: Nästa vecka drar vi igång Rapid Alert-systemet "
                "som finns med i handlingsplanerna också."  # codespell:ignore
            ),
            (
                "SV: Nästa vecka drar vi igång Rapid Alert-systemet "
                "som finns med i handlingsplanerna också."  # codespell:ignore
            ),
        ),
        (
            (
                "RO: Nu uitați să dați like, să lăsați un comentariu și să "
                "distribuiți acest material video pe alte rețele sociale."
            ),
            (
                "RO: Nu uitați să dați like, să lăsați un comentariu și să "
                "distribuiți acest material video pe alte rețele sociale."
            ),
        ),
        (
            (
                "ES: ¿Tiene previsto que este proyecto piloto se "
                "extienda a otros Estados miembros?"
            ),
            (
                "ES: ¿Tiene previsto que este proyecto piloto se "
                "extienda a otros Estados miembros?"
            ),
        ),
    ],
)
def test_basic_punc_cased(test_input, expected):
    normalizer = BasicTextNormalizer(preserve_case=True, preserve_punctuation=True)
    returned = normalizer(test_input)
    assert expected == returned


@pytest.mark.parametrize(
    "lang_code, test_input, expected",
    [
        (
            "nl",
            ("48% van de Europeanen was het daarmee eens."),
            ("48% van de europeanen was het daarmee eens"),
        ),
        (
            "es",
            (
                "¿Tiene previsto que este proyecto piloto se "
                "extienda a otros Estados miembros?"
            ),
            (
                "tiene previsto que este proyecto piloto se "
                "extienda a otros estados miembros"
            ),
        ),
        (
            "sv",
            (
                "Nästa vecka drar vi igång Rapid Alert-systemet "
                "som finns med i handlingsplanerna också."  # codespell:ignore
            ),
            (
                "nästa vecka drar vi igång rapid alert-systemet "
                "som finns med i handlingsplanerna också"  # codespell:ignore
            ),
        ),
    ],
)
def test_normalize_standardize(lang_code, test_input, expected):
    charset = get_allowed_charset([lang_code])
    test_input = lowercase_normalize(test_input, charset=charset, lang_code=lang_code)
    returned = standardize_text((test_input), lang_code=lang_code)
    assert expected == returned


@pytest.mark.parametrize(
    "lang_code, test_input, expected",
    [
        (
            "nl",
            ("48% van de Europeanen was het daarmee eens."),
            ("48% van de europeanen was het daarmee eens."),
        ),
        (
            "es",
            (
                "¿Tiene previsto que este proyecto piloto se "
                "extienda a otros Estados miembros?"
            ),
            (
                "?tiene previsto que este proyecto piloto se "
                "extienda a otros estados miembros?"
            ),
        ),
        (
            "sv",
            (
                "Nästa vecka drar vi igång Rapid Alert-systemet "
                "som finns med i handlingsplanerna också."  # codespell:ignore
            ),
            (
                "nästa vecka drar vi igång rapid alert-systemet "
                "som finns med i handlingsplanerna också."  # codespell:ignore
            ),
        ),
    ],
)
def test_normalize_standardize_punc(lang_code, test_input, expected):
    charset = get_allowed_charset([lang_code])
    test_input = lowercase_normalize(test_input, charset=charset, lang_code=lang_code)
    returned = standardize_text(
        (test_input), preserve_punctuation=True, lang_code=lang_code
    )
    assert expected == returned


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            ("This is a test for currency 100 €"),
            ("This is a test for currency 100 €"),
        ),
        (
            ("This is a test for currency 100€"),
            ("This is a test for currency 100€"),
        ),
        (
            ("This is a test for currency £100"),
            ("This is a test for currency £100"),
        ),
        (
            ("This is a test for currency £ 100"),
            ("This is a test for currency £ 100"),
        ),
        (
            ("This is a test for numbers 89.190"),
            ("This is a test for numbers 89190"),
        ),
        (
            ("This is a test for numbers 18.89,1"),
            ("This is a test for numbers 1889,1"),
        ),
        (
            ("This should remove comma A,B"),
            ("This should remove comma A B"),
        ),
        (
            ("This is a test for ordinals 1100."),
            ("This is a test for ordinals 1100."),
        ),
        (
            ("This is a test for 1100. ordinal"),
            ("This is a test for 1100. ordinal"),
        ),
    ],
)
def test_basic_remove_punc(test_input, expected):
    normalizer = BasicTextNormalizer(preserve_case=True)
    returned = normalizer(test_input)
    assert expected == returned

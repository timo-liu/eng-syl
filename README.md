# English Syllabifier (eng_syl)
This is a GRU-based neural network designed for English word syllabification. The model was trained on data from the  [Wikimorph](https://link.springer.com/chapter/10.1007/978-3-030-78270-2_72) dataset.

## Usage

Use the `syllabify()` function from the `Syllabel` class to syllabify your words:

>     >>> from eng_syl.syllabify import Syllabel
>     >>> syllabler = Syllabel()
>     >>> syllabler.syllabify("chomsky")
>     'chom-sky'

`syllabify()` parameters

 - **text**: *string*- English text to be syllabified. Input should only contain alphabetic characters.

`syllabify()` returns the given word with hyphens inserted at syllable boundaries.

## Onceler (Onset, Nucleus, Coda Segmenter)

The `onc_split()` function from the  `Onceler` class splits single syllables into their constituent Onset, Nucleus, and Coda components.

>     >>> from eng_syl.onceler import Onceler
>     >>> lorax = Onceler()
>     >>> print(lorax.onc_split("sloan")
>     'sl-oa-n'

 - **text**: *string* - English single syllable word/ component to be segmented into Onset, Nucleus, Coda. Input should only contain alphabetic characters.

## Phonify (Grapheme sequence to IPA estimation)

The `ipafy()` function from the  `on_to_phon` class tries to approximate an IPA pronunciation from a sequence of graphemes.

>     >>> from eng_syl.phonify import onc_to_phon
>     >>> skibidi = onc_to_phon()
>     >>> print(skibidi.ipafy(['b', 'u', 'tt'])
>     'bʌt'

 - **sequence**: *array of strings* - sa sequence of English viable onsets, nuclei, and coda


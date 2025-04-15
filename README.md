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
>     >>> onc = Onceler()
>     >>> print(onc.onc_split("schmear")
>     'schm-ea-r'

 - **text**: *string* - English single syllable word/ component to be segmented into Onset, Nucleus, Coda. Input should only contain alphabetic characters.

## Phonify (Grapheme sequence to IPA estimation)

The `ipafy()` function from the  `onc_to_phon` class tries to approximate an IPA pronunciation from a sequence of graphemes.

>     >>> from eng_syl.phonify import onc_to_phon
>     >>> otp = onc_to_phon()
>     >>> print(otp.ipafy(['schm', 'ea', 'r'])
>     ['ʃm', 'ɪ', 'r']

 - **sequence**: *array of strings* - sa sequence of English viable onsets, nuclei, and coda

# 4.0.2 Notes
Fixed a typo in build_model(), where improper shape was being passed into Input()
Reverted class name from Syllabel -> Syllable -> Syllabel

# 4.0.3 Notes
Added handling for non-alpha characters in string; syllabify() won't break immediately if you pass a string like 'he23llotruc38k'. Instead, syllabify() syllabifies the string, ignoring non-alpha characters, and reinserts the non-alpha characters with hyphenation -> 'he23l-lo-truc38k'. This allows for handling of prehyphenated words like 'u-turn' -> 'u--turn'.
Also added an arg for returning the syllables as a list in syllabify(word, return_list = False). Should be capable of handling most strings now.

# 4.0.4 Notes
Added arg save_clean to syllabify(word, save_clean = True). When save_clean, new words will be saved to self.clean for future reference.

# 4.0.7 Notes
Added evaluate_english_validity(syllable) to Syllabel, which returns the onset nucleus coda decomposition split by hyphens if the syllable is likely English pronounceable, and False if unlikely.
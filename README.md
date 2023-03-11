# English Syllabifier (eng_syl)
This program implements a sequence labelling Bidirectional LSTM to identify syllable boundaries in English words. The model was trained on data from the  [WebCelex](http://celex.mpi.nl/) English wordform corpus.

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

 - text: string - English single syllable word/ component to be segmented into Onset, Nucleus, Coda. Input should only contain alphabetic characters.


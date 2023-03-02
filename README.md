# English Syllabifier (eng_syl)
This program implements a sequence labelling Bidirectional LSTM to identify syllable boundaries in English words. The model was trained on data from the  [WebCelex](http://celex.mpi.nl/) English wordform corpus.

Use the `syllabify()` function from the `Syllabel` class to syllabify your words:

>     >>> from eng_syl import Syllabel
>     >>> sylabler = Syllabel()
>     >>> syllabler.syllabify("chomsky")
>     'chom-sky'

`syllabify()` parameters

 - **text**: *string*- English text to be syllabified. Input should only contain alphabetic characters.

`syllabify()` returns the given word with hyphens inserted at syllable boundaries.

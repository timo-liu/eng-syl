title: 'eng-syl: A Python package for phon'
tags:
  - Python
  - Linguistics
  - Syllables
  - Orthography
authors:
  - name: Timothy Liu
    orcid: 0009-0006-9071-0728
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
affiliations:
 - name: University of California Davis, United States
   index: 1
   ror: 05rrcem69
date: 15 April 2025
bibliography: paper.bib

# Summary
    In English, the orthographic syllable is conventionally segmented differently from phonetic syllable rules. CUrrent dictionaries often follow morphological boundaries rather than phonetic boundaries when defining orthographic syllables, making orthography to ipa alignment difficult when examining syllables. eng-syl is an Python package for segmenting orthographic English words and pseudowords according to the maximal onset principle, and for further levels of graphemic analysis. The package is intuitive and lightweight, with parameters for custom syllable dictionaries, grapheme level segmentation, and grapheme to phoneme tools. eng-syl provides functionality for losslessly mapping grapheme to phoneme segments, allowing for deeper analysis of text to sound correspondences.
    
# Statement of Need
    The syllable is a unit of speech construction consisting of a consonant onset, a sonorous nucleus, and an optional consonant coda. Syllables are broadly considered ubiquitous across spoken languages. Contemporary models of spoken word processing consider syllable level information as an important part of spoken word recognition, alongside phonemic, word, and phrase level information (Ortiz-Barajas et al., 2023; Batterink, 2020; Giraud & Poeppel, 2012). Although the syllable is primarily a phonetic construct, it is useful to consider when examining the orthography of languages. Languages such as Cherokee use a syllable based orthography (syllabary), where each icon represents a phonetic syllable in the language. In Chinese, a logographic language, each character represents a syllabic word. In languages with alphabetic orthographies such as French and English, understanding the syllable in text is important for developing literacy (Sherman, 2018; Gallet et al., 2020). In such alphabetic orthographies, however, there is often a disconnect between the orthographic syllable and the phonetic syllable. Phonetic syllables generally follow the maximal onset principle, where consonant phonemes attributable either to a preceding syllable coda or a proceeding syllable onset are attributed to the proceeding onset so long as the language's phonotactic rules are preserved (see 8.7.1 MOP (Maximum Onset Principle) – American English Phonetics, n.d. for examples). Meanwhile, in English, the convention for orthographic syllabification is less rigid. Some words are orthographically syllabified consistent to the maximal onset principle, e.g. "diploma" syllabified as "di.plo.ma" from the Merriam-Webster dictionary. Other words are orthographically syllabified most consistently with morphological boundaries, e.g. "traumatic" syllabified as "trau.ma.tic" from the Merriam-Webster dictionary (which would be syllabified as "trau.ma.tic" if consistent with the maximal onset principle). eng-syl comes with its own dictionary of hand-curated orthographic syllabifications drawn from the CMU pronouncing dictionary and WikiMorph data set, as well as a pretrained neural network for syllabifying out of dictionary words. eng-syl can also divide phonotactically viable English syllables into their constituent grapheme onset, nuclei, and coda.

# References
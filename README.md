# Bridge between ETCBC and Open Scriptures

[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/ETCBC/bridging/)](https://archive.softwareheritage.org/browse/origin/https://github.com/ETCBC/bridging/)
[![DOI](https://zenodo.org/badge/116673254.svg)](https://zenodo.org/badge/latestdoi/116673254)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

[![etcbc](programs/images/etcbc.png)](http://www.etcbc.nl)
[![dans](programs/images/dans.png)](https://dans.knaw.nl/en)
[![tf](programs/images/tf-small.png)](https://annotation.github.io/text-fabric/tf)

### BHSA Family

* [bhsa](https://github.com/etcbc/bhsa) Core data and feature documentation
* [phono](https://github.com/etcbc/phono) Phonological representation of Hebrew words
* [parallels](https://github.com/etcbc/parallels) Links between similar verses
* [valence](https://github.com/etcbc/valence) Verbal valence for all occurrences
  of some verbs
* [trees](https://github.com/etcbc/trees) Tree structures for all sentences
* [bridging](https://github.com/etcbc/bridging) Open Scriptures morphology
  ported to the BHSA
* [pipeline](https://github.com/etcbc/pipeline) Generate the BHSA and SHEBANQ
  from internal ETCBC data files
* [shebanq](https://github.com/etcbc/shebanq) Engine of the
  [shebanq](https://shebanq.ancient-data.org) website

### Extended family

* [dss](https://github.com/etcbc/dss) Dead Sea Scrolls
* [extrabiblical](https://github.com/etcbc/extrabiblical)
  Extra-biblical writings from ETCBC-encoded texts
* [peshitta](https://github.com/etcbc/peshitta)
  Syriac translation of the Hebrew Bible
* [syrnt](https://github.com/etcbc/syrnt)
  Syriac translation of the New Testament

## About

Both the BHSA and the
[OpenScriptures](https://github.com/openscriptures/morphhb)
add linguistic markup to the Hebrew Bible.

The BHSA is the product of years of encoding work by researchers,
in a strongly algorithmic fashion,
although not without human decisions at the micro level.

OpenScriptures represents a crowd sourced approach.

In this repo we compare them.
As a by-product, a mapping of BHSA words to OS morphemes is made,
and the OS morphological information of each word is made available
as a new word feature in the BHSA.

When we first made the comparison, in 2017,
only 88% of the OpenScriptures Morphology was completed.

In 2021 we have pulled the same repository again, used a new version of the BHSA
and did the comparison again.

# Author

[Dirk Roorda](https://github.com/dirkroorda)


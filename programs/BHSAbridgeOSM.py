#!/usr/bin/env python
# coding: utf-8

# <img align="right" src="images/dans-small.png"/>
# <img align="right" src="images/tf-small.png"/>
# <img align="right" src="images/etcbc.png"/>

# # BHSA and OpenScriptures bridge
# 
# Both the BHSA and the OpenScriptures represent efforts to add linguistic markup to the Hebrew Bible.
# 
# The BHSA is the product of years of encoding work by researchers, in a strongly algorithmic fashion, although
# not without human decisions at the micro level.
# 
# OpenScriptures represents a crowd-sourced approach.
# 
# Regardless of theoretical considerations on the validity of these approaches, it is worthwhile to be able to compare them.
# Moreover, for some research problems, it might be helpful to use both encodings in one toolkit.
# 
# In this repo we develop a way of doing exactly this.
# 
# We make a link between the morphology in the
# [Openscriptures](http://openscriptures.org)
# and the linguistics in the [BHSA](https://github.com/ETCBC/bhsa).
# 
# We proceed as follows:
# 
# * extract the morphology from the files in
#   [openscriptures/morphhb/wlc](https://github.com/openscriptures/morphhb/tree/master/wlc)
# * link the words in the openscripture files to slots in the BHSA
# * compile the openscripture morphology data into a TF feature file.
# 
# With this in hand, we have the OpenScriptures morphology in Text-Fabric, aligned to the BHSA.
# That opens the way for further comparisons, which take the actual morphology into account.

# ## History
# 
# When we first made the comparison, in 2017, only 88% of the OpenScriptures Morphology was fixed.
# 
# In 2021 we have pulled the same repository of Open Scriptures again, and used a new version of the BHSA as well.
# It turns out the 100% of the words have been morphologically annotated by OpenScriptures now.

# ## Application
# 
# This notebook sets the stage for focused comparisons between the BHSA features on words and the OSM morphology.
# 
# See
# 
# * [category](category.ipynb)
# * [language](language.ipynb)
# * [part-of-speech](part-of-speech.ipynb)
# * [verb](verb.ipynb)

# In[1]:


import os
import sys
import collections
import yaml
from glob import glob
from lxml import etree
from itertools import zip_longest
from functools import reduce
from unicodedata import normalize, category

from tf.fabric import Fabric
from tf.core.helpers import rangesFromSet, formatMeta
import utils


# # Pipeline
# See [operation](https://github.com/ETCBC/pipeline/blob/master/README.md#operation)
# for how to run this script in the pipeline.

# In[2]:


if "SCRIPT" not in locals():
    SCRIPT = False
    FORCE = True
    CORE_NAME = "bhsa"
    NAME = "bridging"
    VERSION = "2021"


def stop(good=False):
    if SCRIPT:
        sys.exit(0 if good else 1)


# This notebook can run a lot of tests and create a lot of examples.
# However, when run in the pipeline, we only want to create the two `osm` features.
# 
# So, further on, there will be quite a bit of code under the condition `not SCRIPT`.

# # Setting up the context: source file and target directories
# 
# The conversion is executed in an environment of directories, so that sources, temp files and
# results are in convenient places and do not have to be shifted around.

# In[3]:


repoBase = os.path.expanduser("~/github/etcbc")
coreRepo = "{}/{}".format(repoBase, CORE_NAME)
thisRepo = "{}/{}".format(repoBase, NAME)

coreTf = "{}/tf/{}".format(coreRepo, VERSION)

thisTemp = "{}/_temp/{}".format(thisRepo, VERSION)
thisTempTf = "{}/tf".format(thisTemp)

thisTf = "{}/tf/{}".format(thisRepo, VERSION)


# # Test
# 
# Check whether this conversion is needed in the first place.
# Only when run as a script.

# In[4]:


if SCRIPT:
    (good, work) = utils.mustRun(
        None, "{}/.tf/{}.tfx".format(thisTf, "osm"), force=FORCE
    )
    if not good:
        stop(good=False)
    if not work:
        stop(good=True)


# # Load the BHSA data

# In[5]:


utils.caption(4, "Load the existing TF dataset")
TF = Fabric(locations=coreTf, modules=[""])

api = TF.load(
    """
    book
    g_cons_utf8 g_word_utf8
"""
)
api.makeAvailableIn(globals())


# # Reading Open Scriptures Morphology

# In[6]:


NB_DIR = os.getcwd()
OS_BASE = os.path.expanduser("~/github/openscriptures/morphhb/wlc")
os.chdir(OS_BASE)


# ## Mapping the book names
# OSM uses abbreviated book names.
# We map them onto the (latin) book names of the BHSA.
# 
# Here is a list of the BHSA books.

# In[7]:


bhsBooks = [F.book.v(n) for n in F.otype.s("book")]
utils.caption(0, " ".join(bhsBooks))


# The next cell can be used to retrieve the OSM book names,
# from which the ordered list `osmBooks` below can be composed manually.

# In[8]:


osmBookSet = set(fn[0:-4] for fn in glob("*.xml") if fn != "VerseMap.xml")
utils.caption(0, " ".join(sorted(osmBookSet)))


# We list the books in the "canonical" order (as given in the BHSA).

# In[9]:


osmBooks = """
Gen Exod Lev Num Deut
Josh Judg 1Sam 2Sam 1Kgs 2Kgs
Isa Jer Ezek Hos Joel Amos Obad
Jonah Mic Nah Hab Zeph Hag Zech Mal
Ps Job Prov Ruth Song Eccl Lam Esth
Dan Ezra Neh 1Chr 2Chr
""".strip().split()


# We check whether we did not overlook books or missed changes in the OSM abbreviations of the books.

# In[10]:


osmBookSet == set(osmBooks)


# Now we can construct the mapping, both ways.

# In[11]:


osmBookFromBhs = {}
bhsBookFromOsm = {}
for (i, bhsBook) in enumerate(bhsBooks):
    osmBook = osmBooks[i]
    osmBookFromBhs[bhsBook] = osmBook
    bhsBookFromOsm[osmBook] = bhsBook


# ## Consonantal matters
# 
# For alignment purposes, we reduce all textual material to its consonantal representation.
# Sometimes we need to blur the distinction between final consonants and their normal counterparts.
# 
# In order to strip consonants from all their diacritical marks, we use unicode denormalization and
# unicode character categories.

# In[12]:


NS = "{http://www.bibletechnologies.net/2003/OSIS/namespace}"
NFD = "NFD"
LO = "Lo"

finals = {
    "ך": "כ",
    "ם": "מ",
    "ן": "נ",
    "ף": "פ",
    "ץ": "צ",
}

finalsI = {v: k for (k, v) in finals.items()}


# `toCons(s)`: strip all pointing (accents, vowels, dagesh, s(h)in dot) from all characters in string `s`.
# 
# `final(c)`: replace consonant `c` by its final counterpart, if there is one, otherwise return `c`.
# 
# `finalCons(s)`: replace the last character of `s` by its final counterpart.
# 
# `unFinal(s)`: replace all consonants in `s` by their non-final counterparts.

# In[13]:


def toCons(s):
    return "".join(c for c in normalize(NFD, s) if category(c) == LO)


def final(c):
    return finalsI.get(c, c)


def finalCons(s):
    return s[0:-1] + final(s[-1])


def unFinal(s):
    return "".join(finals.get(c, c) for c in s)


# ## Read OSM books
# 
# We are going to read the OSM files.
# They correspond to books.
# 
# We drill down to verse nodes and pick up the `<w>` elements.
# What we need from these elements is the full text content and the attributes
# `lemma` and `morph`.
# 
# We ignore markup within the full text content of the `<w>` elements.
# 
# The material we extract, may contain `/`.
# We split the text content and the `lemma` and `morph` content on `/`, and recombine the resulting parts in
# OSM morpheme entries, having each a full-text bit, a morph bit and a lemma bit.
# 
# Caveat: when splitting the morpheme string, we should first split off the first character, which indicates language,
# and then add it to all the parts!
# 
# So, one `<w>` element may give rise to several morpheme entries.
# 
# The full text is fully pointed. We also compute a consonantal version of the full text and store it
# within the morpheme entries.
# 
# We end up with a list, `osmMorphemes` of morpheme entries.
# 
# In passing, we count the `<w>` elements without `morph` attributes, and those without textual content.
# 
# We also store the book, chapter, verse and sequence number of the `<w>` element in each entry.

# In[14]:


def readOsmBook(osmBook, osmMorphemes, stats):
    infile = "{}.xml".format(osmBook)
    parser = etree.XMLParser(remove_blank_text=True, ns_clean=True)
    root = etree.parse(infile, parser).getroot()
    osisTextNode = root[0]
    divNode = osisTextNode[1]
    chapterNodes = list(divNode)
    utils.caption(
        0,
        "reading {:<5} ({:<15}) {:>3} chapters".format(
            osmBook, bhsBookFromOsm[osmBook], len(chapterNodes)
        ),
    )
    ch = 0
    for chapterNode in chapterNodes:
        if chapterNode.tag != NS + "chapter":
            continue
        ch += 1
        vs = 0
        for verseNode in list(chapterNode):
            if verseNode.tag != NS + "verse":
                continue
            vs += 1
            w = 0
            for wordNode in list(verseNode):
                if wordNode.tag != NS + "w":
                    continue
                w += 1
                lemma = wordNode.get("lemma", None)
                morph = wordNode.get("morph", None)
                text = "".join(x for x in wordNode.itertext())

                lemmas = lemma.split("/") if lemma is not None else []
                morphs = morph.split("/") if morph is not None else []
                if len(morphs) > 1:
                    lang = morphs[0][0]
                    morphs = [morphs[0]] + [lang + m for m in morphs[1:]]
                texts = text.split("/") if text is not None else []
                # zip_longest accomodates for unequal lengths of its operands
                # for missing values we fill in ''
                for (lm, mph, tx) in zip_longest(lemmas, morphs, texts, fillvalue=""):
                    txc = None if tx is None else toCons(tx)
                    osmMorphemes.append((tx, txc, mph, lm, osmBook, ch, vs, w))
                    if not mph:
                        stats["noMorph"] += 1
                    if not tx:
                        stats["noContent"] += 1


# That was the definition of the read function, now we are going to execute it.

# In[15]:


osmMorphemes = []
stats = dict(noMorph=0, noContent=0)

for bn in F.otype.s("book"):
    bhsBook = T.sectionFromNode(bn, lang="la")[0]
    osmBook = osmBookFromBhs[bhsBook]
    readOsmBook(osmBook, osmMorphemes, stats)

utils.caption(
    0,
    """
BHS words:       {:>6}
OSM Morphemes:   {:>6}
No morphology:   {:>6}
No content:      {:>6}
{} % of the words are morphologically annotated.
""".format(
        F.otype.maxSlot,
        len(osmMorphemes),
        stats["noMorph"],
        stats["noContent"],
        round(
            100
            * (len(osmMorphemes) - stats["noMorph"] - stats["noContent"])
            / len(osmMorphemes)
        ),
    ),
)


# To give an impression of the contents of this list, we show the first few members.
# The column specification is:
# 
#     consonantal fully-pointed morph lemma book chapter verse w-number

# In[16]:


list(osmMorphemes[0:15])


# # Alignment
# 
# We now have to face the task to map the BHSA words to the OSM morphemes.
# 
# We will encounter the challenge that at some spots the consonantal contents of the WLC (the source of the OSM)
# is different from that of the BHS, the source of the BHSA.
# 
# Another challenge is that at some points the analysis behind the OSM differs from that of the BHSA in such a way
# that the BHSA has a word-split within an OSM morpheme.
# 
# Yet another source of problems is that the BHSA inserts "empty" articles in places where the pointing in the
# surrounding material allows to conclude that an article is present, although it does not have a consonantal presence anymore.

# We need a function to quickly show what is going on in difficult spots.
# 
# `showCase(w, j, ln)` shows the BHSA from word `w` onwards, and the OSM from morpheme `j` onwards.
# It lists `ln` positions in both sources.

# In[17]:


def showCase(w, j, ln):
    print(T.sectionFromNode(w))
    print("BHS")
    for n in range(w, w + ln):
        print("word  {} = [{}]".format(n, toCons(F.g_cons_utf8.v(n))))
    print("OSM")
    for n in range(j, j + ln):
        print("morph {} = [{}]".format(n, osmMorphemes[n][1]))


# We also define another function to easy inspect difficult spots.
# 
# `BHSvsOSM(ws, js)` compares the BHSA words specified by list `ws` with the OSM morphemes
# specified by list `js`.
# 
# Here we bump into the fact that the BHSA deals with whole words, and the OSM splits into morphemes.
# In this case, the pronominal suffix is treated as a separate morpheme.

# In[18]:


def BHSvsOSM(ws, js):
    print(
        "{}\n{:<25}BHS {:<30} = {}\n{:<25}OSM {:<30} = {}".format(
            "{} {}:{}".format(*T.sectionFromNode(ws[0])),
            " ",
            ", ".join(str(w) for w in ws),
            "/".join(F.g_word_utf8.v(w) for w in ws),
            " ",
            ", ".join("w{}".format(osmMorphemes[j][7]) for j in js),
            "/".join(osmMorphemes[j][0] for j in js),
        )
    )


# ## Algorithm
# 
# We have to develop a way of aligning each BHS word with one or more OSM morphemes.
# 
# For each BHS word, we grab OSM morphemes until all consonants in the BHS word have been matched.
# If needed, we grab additional BHS words
# when the current OSM string happens to be longer than the current BHS word.
# 
# We will encounter cases where this method breaks down: exceptions.
# We will collect them for later inspection.
# 
# The exceptions are coded as follows:
# 
# If `w: n` is in the dictionary of exceptions, it means that slot (word) `w` in the BHSA is different from its counterpart morpheme(s) in the OSM.
# 
# If `n > 0`, that many OSM morphemes will be gobbled to align with slot `w`.
# 
# If `n < 0`, that many slots from `w` will be gobbled to match the current OSM morpheme.
# 
# There are various subtleties involved, see the inline content in the code below.

# In[19]:


allExceptions = {
    "2017": {
        215253: 1,
        266189: 1,
        287360: 2,
        376865: 1,
        383405: 2,
        384049: 1,
        384050: 1,
        405102: -2,
    },
    "2021": {
        215256: 1,
        266192: 1,
        287363: 2,
        376869: 1,
        383409: 2,
        384053: 1,
        384054: 1,
        405108: -2,
    },
}

exceptions = allExceptions[VERSION]


# In[20]:


# index in the osmMorphemes list
j = -1

# mapping from BHSA slot numbers to OSM morphemes indices
osmFromBhs = {}

u = None
remainingErrors = False
for w in F.otype.s("word"):
    # the previous iteration may have already dealt with this word
    # in that case, we skip to the next word
    # the signal is: w <= u
    if u is not None and w <= u:
        continue

    # we get the consonantal BHSA word string
    bhs = toCons(F.g_cons_utf8.v(w))

    # if the BHSA word is empty, we do not link it to any OSM morpheme
    # and continue
    if bhs == "":
        continue

    # we are going to collect OSM morphemes
    # as long as the consonantal reps of the morpheme fit into the BHSA word
    j += 1
    startJ = j
    startW = w
    osm = osmMorphemes[j][1]

    # but if the word is listed as exception, we collect as many morphemes
    # as specified in the exception
    maxGobble = exceptions.get(w, None)
    gobble = 1
    while (len(osm) < len(bhs) and bhs.startswith(osm)) or (
        maxGobble is not None and maxGobble > 0
    ):
        if maxGobble is not None and gobble >= maxGobble:
            break
        j += 1
        osm += osmMorphemes[j][1]
        gobble += 1

    # if the OSM morphemes have become longer than the BHSA word,
    # we eat up the following BHSA word(s)
    # we let u hold the new BHSA word position
    u = w
    gobble = 1
    while (len(osm) > len(bhs) and osm.startswith(bhs)) or (
        maxGobble is not None and maxGobble < 0
    ):
        if maxGobble is not None and gobble >= -maxGobble:
            break
        u += 1
        bhs += toCons(F.g_cons_utf8.v(u))
        gobble += 1
    gobble = 1

    # if the BHSA words exceed the OSM morphemes found so far, we draw in additional OSM morphemes
    # (for the last time)
    while len(osm) < len(bhs) and bhs.startswith(osm):
        if maxGobble is not None and gobble >= maxGobble:
            break
        j += 1
        osm += osmMorphemes[j][1]
        gobble += 1

    # now we have gathered a BHSA string of material, and an OSM string of material
    # We test if both strings are equal (modulo final consonant issues)
    # If not: alignment breaks down, we stop the loop and show the offending case.
    # The programmer should inspect the case and add an exception.
    if maxGobble is None and finalCons(bhs) != finalCons(osm):
        utils.caption(
            0,
            """Mismatch in {} at BHS-{} OS-{}->{}:\nbhs=[{}]\nos=[{}]""".format(
                "{} {}:{}".format(*T.sectionFromNode(w)),
                w,
                startJ,
                j,
                bhs,
                osm,
            ),
        )
        showCase(w - 5, startJ - 5, j - startJ + 10)
        remainingErrors = True
        break

    # but if all is well, we link the BHSA words in question to the OSM morphemes in question
    # If the BHSA string contains multiple words, we link all those words to all morphemes
    for k in range(startW, u + 1):
        for m in range(startJ, j + 1):
            osmFromBhs.setdefault(k, []).append(m)

if not remainingErrors:
    utils.caption(0, "Succeeded in aligning BHS with OSM")
    utils.caption(
        0,
        "{} BHS words matched against {} OSM morphemes with {} known exceptions".format(
            len(osmFromBhs),
            len(osmMorphemes),
            len(exceptions),
        ),
    )


# We have constructed in passing the mapping `osmFromBhs`,
# which maps BHSA words onto corresponding sequences of OSM morphemes.
# We also compute the inverse of this, `bhsFromOsm`.

# In[21]:


# mapping from OSM morphemes (by index in osmMorphemes list) to BHSA slot numbers
# It is the inverse of osmFromBhs
bhsFromOsm = {}

for (w, js) in osmFromBhs.items():
    for j in js:
        bhsFromOsm.setdefault(j, []).append(w)
utils.caption(0, "{} morphemes mapped in bhsFromOsm".format(len(bhsFromOsm)))


# # Inspection of problems
# 
# We have encountered irregularities, but we want to make sure we have seen all potential
# alignment problems.
# We do this by adding a sanity check: find all cases where
# the consonantal material in a BHSA word is not the
# concatenation of the consonantal material in in its OSM morphemes.
# 
# We have now several irregularities to inspect.
# 
# 1. **Multiplicity**
#    * By looking into `bhsFromOsm` we can find the OSM morphemes that contain consonantal material
#      from multiple BHSA words.
#      These are interesting points of difference between the BHSA and OSM encoding, because
#      in these cases the OSM produces other word/morpheme boundaries than the BHSA.
#    * We also inspect cases where a BHSA word corresponds to more than two OSM morphemes.
# 1. **Consonantal Sanity**
#    By analysis after the fact, we gather all consonantal discrepancies
# 1. **Exceptions**
#    While developing the algorithm, we needed to invoke a small number of manual exceptions.
# 
# 
# Now we want to make a comprehensive list of all problematic cases encountered during
# alignment.
# 
# We will add the BHSA word numbers involved in a problematic case to the set `problematic`.
# When we proceed to compare morphology, we will exclude the problematic cases.

# In[22]:


problematic = set()


# ## Multiplicity
# We gather the cases of multiple BHSA words against a single OSM morpheme.

# In[23]:


multipleOSM = {}  # OSM morphemes in correspondence with multiple BHS slots
noOSM = {}  # OSM morphemes that do not correspond to any BHSA word

countMultipleOSM = (
    collections.Counter()
)  # how many times n BHSA words are linked to the same OSM morpheme

for (j, ws) in bhsFromOsm.items():
    nws = len(ws)
    if nws > 1:
        multipleOSM[j] = nws
        countMultipleOSM[nws] += 1
    elif nws == 0:
        noOSM.add(j)

utils.caption(
    0,
    "OSM morphemes without corresponding BHSA word:                {:>5}".format(
        len(noOSM)
    ),
)
utils.caption(
    0,
    "OSM morphemes corresponding to multiple BHSA words:           {:>5}".format(
        len(multipleOSM)
    ),
)
for (nws, amount) in sorted(countMultipleOSM.items()):
    utils.caption(
        0,
        "OSM morphemes corresponding to {} BHSA words:                  {:>5}".format(
            nws, amount
        ),
    )


# In[24]:


for j in multipleOSM:
    ws = bhsFromOsm[j]
    problematic |= set(ws)
    if not SCRIPT:
        BHSvsOSM(ws, [j])


# ## Consonantal sanity
# Which non-empty BHSA words are not the concatenation of their OSM morphemes?
# 
# We do not consider the cases where more than one BHSA word corresponds to an OSM morpheme,
# because we have already gathered those cases above.

# In[25]:


insaneBHS = set()  # alignment problems by BHSA slot number
insaneOSM = set()  # alignment problems by OSM morpheme index in osmMorphemes

# We compute the slot numbers of that are part of a multiple slot alignment to a morpheme
multipleBHS = reduce(set.union, (bhsFromOsm[j] for j in multipleOSM), set())

# Gather the insanities
for (w, js) in osmFromBhs.items():
    if w in multipleBHS:
        continue
    cw = toCons(F.g_cons_utf8.v(w))
    cjs = "".join(osmMorphemes[j][1] for j in js)
    if unFinal(cw) != unFinal(cjs):
        insaneBHS.add(w)
        insaneOSM |= set(js)
utils.caption(0, "insane BHS words:     {:>4}".format(len(insaneBHS)))
utils.caption(0, "insane OSM morphemes: {:>4}".format(len(insaneOSM)))


# In[26]:


for w in sorted(insaneBHS):
    problematic.add(w)
    js = osmFromBhs[w]
    if not SCRIPT:
        BHSvsOSM([w], js)


# ## More than two morphemes per word
# 
# Let's study the mapping of BHSA words to OSM morphemes in a bit more detail.
# We are interested in the question: to how many morphemes can words map?
# 
# Later we shall see that we can deal with 1 and 2 morphemes per word.
# 
# We deem words that map to more than two morphemes problematic.
# 
# This turns out to be a very small minority.

# In[27]:


morphemesPerWord = collections.Counter()
tooMany = set()
for (w, js) in osmFromBhs.items():
    n = len(js)
    morphemesPerWord[n] += 1
    if n > 2:
        tooMany.add(w)

for (ln, amount) in sorted(morphemesPerWord.items()):
    utils.caption(0, "{:>2} morphemes per word: {:>6}".format(ln, amount))


# In[28]:


for w in sorted(tooMany):
    js = osmFromBhs[w]
    if not SCRIPT:
        BHSvsOSM([w], js)
    problematic.add(w)


# ## Exceptions
# 
# Finally, we inspect the cases that correspond to the manual exceptions.

# In[29]:


for (w, n) in exceptions.items():
    if n > 0:
        js = osmFromBhs[w]
        if not SCRIPT:
            BHSvsOSM([w], js)
        problematic.add(w)
    else:
        j = osmFromBhs[w][0]
        ws = bhsFromOsm[j]
        if not SCRIPT:
            BHSvsOSM(ws, [j])
        problematic |= set(ws)


# Here is the number of problematic words in the BHSA that we will exclude from comparisons.

# In[30]:


utils.caption(
    0, f"There are {len(problematic)} problematic words in the BHSA wrt to OSM"
)
utils.caption(0, "These will be excluded from further comparisons")


# # Missing morphology
# 
# We make a list of word nodes for which no morpheme has been tagged with morphology.
# Only for non-empty words.

# In[31]:


noMorphWords = set()
for w in F.otype.s("word"):
    if not F.g_word_utf8.v(w):
        continue
    hasMorph = False
    for j in osmFromBhs.get(w, []):
        if osmMorphemes[j][2]:
            hasMorph = True
            break
    if not hasMorph:
        noMorphWords.add(w)


# In[32]:


if len(noMorphWords):
    utils.caption(0, f"No OSM morphology for {len(noMorphWords)} non-empty BHSA words")
else:
    utils.caption(0, "There is OSM morphology for all non-empty BHSA words")


# Let's get a feeling for how the non-tagged morphemes are distributed.
# First we represent them as a list of intervals, using a utility function of TF,
# and then we get an overview of the lengths of the intervals.

# In[33]:


noMorphIntervals = rangesFromSet(noMorphWords)

noMorphLengths = collections.Counter()

for interval in noMorphIntervals:
    noMorphLengths[interval[1] - interval[0] + 1] += 1

if noMorphLengths:
    utils.caption(0, "Non-marked-up stretches having length x: y times")
    for (ln, amount) in sorted(noMorphLengths.items()):
        utils.caption(0, "{:>4}: {:>5}".format(ln, amount))
else:
    utils.caption(0, "no non-marked-up stretches")


# # Data generation
# 
# We now proceed to compile the OSM morphology into Text-Fabric features.
# 
# The basic idea is: create a feature `osm` and for each BHSA word, let it contain the contents of the
# corresponding `morph` attribute in the OSM source.
# 
# There are several things to deal with, or not to deal with.
# 
# ## Problematic cases
# We will ignore the problematic cases. More precisely, whenever a BHSA word belongs to a problematic case,
# as diagnosed before, we fill its `osm` feature with the value `*`.
# 
# ## Multiplicity of morphemes
# There are BHSA words that do not correspond to OSM morphemes. The empty words. We will not give them an `osm` value.
# 
# There are BHSA words that correspond to more than two morphemes. We have added them to our problematic list.
# 
# The vast majority of BHSA words correspond to a single OSM morpheme.
# The `osm` feature of those words will be filled
# with the `morph` attribute part of the corresponding OSM morpheme. No problem here.
# 
# The remaining cases consist of BHSA words that correspond to exactly two morphemes.
# We will use the value of the `morph` of the first morpheme to fill the `osm` feature for such words,
# and we will make a new feature, `osm_sf` and fill it with the `morph` of the second morpheme.
# 
# So, we will create a TF module consisting of two features: `osm` and `osm_sf` (osm suffix).
# 
# Let's assemble the feature data.

# In[34]:


osmData = {}
osm_sfData = {}
for (w, js) in osmFromBhs.items():
    if w in problematic:
        osmData[w] = "*"
        continue
    osmData[w] = osmMorphemes[js[0]][2]
    if len(js) > 1:
        osm_sfData[w] = osmMorphemes[js[1]][2]


# In[35]:


genericMetaPath = f"{thisRepo}/yaml/generic.yaml"
bridgingMetaPath = f"{thisRepo}/yaml/bridging.yaml"

with open(genericMetaPath) as fh:
    genericMeta = yaml.load(fh, Loader=yaml.FullLoader)
    genericMeta["version"] = VERSION
with open(bridgingMetaPath) as fh:
    bridgingMeta = formatMeta(yaml.load(fh, Loader=yaml.FullLoader))

metaData = {"": genericMeta, **bridgingMeta}


# In[36]:


nodeFeatures = dict(osm=osmData, osm_sf=osm_sfData)

for f in nodeFeatures:
    metaData[f]["valueType"] = "str"


# And combine it with a bit of metadata.

# In[37]:


utils.caption(4, "Writing tree feature to TF")
TFw = Fabric(locations=thisTempTf, silent=True)
TFw.save(nodeFeatures=nodeFeatures, edgeFeatures={}, metaData=metaData)


# # Diffs
# 
# Check differences with previous versions.

# In[37]:


utils.checkDiffs(thisTempTf, thisTf, only=set(nodeFeatures))


# # Deliver
# 
# Copy the new TF features from the temporary location where they have been created to their final destination.

# In[38]:


utils.deliverDataset(thisTempTf, thisTf)


# # Compile TF

# In[39]:


utils.caption(4, "Load and compile the new TF features")


# In[40]:


TF = Fabric(locations=[coreTf, thisTf], modules=[""])
api = TF.load("language " + " ".join(nodeFeatures))
api.makeAvailableIn(globals())


# In[42]:


utils.caption(4, "Basic tests")
utils.caption(4, "Language according to BHSA and OSM")


# In[54]:


langBhsFromOsm = dict(A="Aramaic", H="Hebrew")
langOsmFromBhs = dict((y, x) for (x, y) in langBhsFromOsm.items())

xLanguage = set()
strangeLanguage = collections.Counter()

for w in F.otype.s("word"):
    osm = F.osm.v(w)
    if osm is None or osm == "" or osm == "*":
        continue
    osmLanguage = osm[0]
    trans = langBhsFromOsm.get(osmLanguage, None)
    if trans is None:
        strangeLanguage[osmLanguage] += 1
    else:
        if langBhsFromOsm[osm[0]] != F.language.v(w):
            xLanguage.add(w)

if strangeLanguage:
    utils.caption(0, "Strange languages")
    for (ln, amount) in sorted(strangeLanguage.items()):
        utils.caption(0, "Strange language {}: {:>5}x".format(ln, amount))
else:
    utils.caption(
        0, "No other languages encountered than {}".format(", ".join(langBhsFromOsm))
    )
utils.caption(0, "Language discrepancies: {}".format(len(xLanguage)))
for w in sorted(xLanguage):
    passage = "{} {}:{}".format(*T.sectionFromNode(w))
    utils.caption(0, f"{passage} word {w}: {F.g_word_utf8.v(w):>12} - BHSA: {F.language.v(w)}; OSM: {langBhsFromOsm[F.osm.v(w)[0]]}")


# # End of pipeline
# 
# If this notebook is run with the purpose of generating data, this is the end then.
# 
# After this tests and examples are run.

# In[41]:

# In[55]:


if SCRIPT:
    stop(good=True)


# Now you can write notebooks to process BHSA data and grab the OSM morphology as you go, like so:
# 
# ```
# A = use("bhsa", mod="etcbc/bridging/tf", hoist=globals())
# ```

# # Tests and examples
# 
# Before we flesh out the alignment algorithm,
# let's find the first point where BHSA and OSM diverge.

# In[56]:


for (i, w) in enumerate(F.otype.s("word")):
    bhs = toCons(F.g_cons_utf8.v(w))
    osm = osmMorphemes[i][1]
    if bhs != osm:
        utils.caption(
            0, "Mismatch at BHS-{} OSM-{}: bhs=[{}] osm=[{}]".format(w, i, bhs, osm)
        )
        break


# In[57]:


showCase(62, 61, 5)


# This is a case of an empty article in the BHSA.
# Let's circumvent this, and move on.

# In[58]:


j = -1
for w in F.otype.s("word"):
    bhs = toCons(F.g_cons_utf8.v(w))
    if bhs == "":
        continue
    j += 1
    osm = osmMorphemes[j][1]
    if bhs != osm:
        utils.caption(
            0,
            """Mismatch at BHS-{} OSM-{}:\nbhs=[{}]\nos=[{}]""".format(w, j, bhs, osm),
        )
        break


# In[59]:


showCase(194, 187, 5)


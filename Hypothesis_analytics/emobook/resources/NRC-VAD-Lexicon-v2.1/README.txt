
THE NRC VALENCE, AROUSAL, AND DOMINANCE LEXICON (aka THE NRC VAD LEXICON)
-------------------------------------------------------------------------

Version: 	2.1
Released: 	March 2025
Copyright: 	2025 National Research Council Canada (NRC)
Created By: Dr. Saif M. Mohammad (Principal Research Scientist, National Research Council Canada)

README Last Updated: March 2025

Home Page:  http://saifmohammad.com/WebPages/nrc-vad.html

Contact:  	Dr. Saif M. Mohammad
          	saif.mohammad@nrc-cnrc.gc.ca
          	uvgotsaif@gmail.com

TABLE OF CONTENTS

	I. 	  General Description
	II.   What is Included: Forms of the Lexicon, Files, And Format
	III.  Relevant Research Papers 
	IV.   Python Code To Analyze Emotions In Text
	V.    NRC VAD Lexicon In Various Languages
	VI.   Using Polar Terms Only, Rescaling, Lemmatization, etc.
	VII.  Version Information And Change Log
	VIII. Other Emotion Lexicons
	IX.   Terms Of Use
	X.    Ethical Considerations
 

I. GENERAL DESCRIPTION
----------------------

Words play a central role in language and thought. Several influential factor analysis
studies have shown that the primary dimensions of word meaning are valence, arousal, and
dominance (VAD). 
- valence is the positive--negative or pleasure--displeasure dimension; 
- arousal is the excited--calm or active--passive dimension; and 
- dominance (aka competence) is the powerful--weak, competent--incompetent, and or 'have
control'--'have no control' dimension.

The NRC Valence, Arousal, and Dominance (VAD) Lexicon v1 includes a list of more than
20,000 English words and their valence, arousal, and dominance scores.  With version 2 we
added an additional ~35,000 terms (~25k unigrams and ~10k multi-word expressions). The
lexicon is markedly larger than any of the existing VAD lexicons. 

For a given term and a dimension (V/A/D), the scores range from -1 (lowest V/A/D) to 1
(highest V/A/D).  The lexicon with its fine-grained real-valued scores was created by
manual annotation. We also show that the ratings obtained are substantially more reliable
than those in existing lexicons. (See associated papers for details.)

Applications: The NRC VAD Lexicon has a broad range of applications in Computational
Linguistics, Psychology, Digital Humanities, Computational Social Sciences, and beyond.
Notably it can be used to:
- study how people use words to convey emotions.
- study how emotions are conveyed through literature, stories, and characters.
- develop better sentiment and emotion detection systems.
- evaluate automatic methods of determining V, A, and D (using NRC VAD entries as gold/reference scores).
- study psychological models of emotions.
- study the role of high VAD words in high emotion intensity sentences, tweets, snippets from literature.
- study stereotypes using the V--D dimensions or warmth--competence dimensions 
  (see the paper 'Words of Warmth: Trust and Sociability Norms for over 26k English Words')

An Interactive Visualization of the NRC VAD Lexicon is available here: 
http://saifmohammad.com/WebPages/nrc-vad.html#Viz

Companion lexicons:
- NRC Emotion Lexicon
- NRC Emotion Intensity Lexicon
- Words of Warmth (the Warmth, Sociability, Trust, and Competence Lexicons)
- WorryWords (the Word--Anxiety Association Lexicon)

Available here:
http://saifmohammad.com/WebPages/lexicons.html

This study was approved by the NRC Research Ethics Board (NRC-REB) under protocol number
2017-98. REB review seeks to ensure that research projects involving humans as
participants meet Canadian standards of ethics.

v1: The lexicon with its fine-grained real-valued scores was created by manual annotation
using best--worst scaling.  The lexicon is markedly larger than any of the existing VAD
lexicons. We also show that the ratings obtained are substantially more reliable than
those in existing lexicons. (See associated paper for details.)


v2: The additional entries for v2 were created using a Likert Rating scale. We also show
that the ratings obtained are highly reliable. (See associated paper for details.)


II. WHAT IS INCLUDED: FORMS OF THE LEXICON, FILES, AND FORMAT
-------------------------------------------------------------

1. NRC-VAD-Lexicon-v2.1.txt: This is the main lexicon file with entries for ~55,000
English words and multi-word expressions. It has four columns (separated by tabs):

- term: The English word for which V, A, and D scores are provided. The words are listed in alphabetic order.
- valence: valence score of the word
- arousal: arousal score of the word
- dominance: dominance score of the word

2. The directory 'OneFilePerDimension' has the same information as in
NRC-VAD-Lexicon-v2.1.txt, but in multiple files -- one for each dimension:

- valence-NRC-VAD-Lexicon-v2.1.txt: Includes valence scores. The words are sorted in decreasing order of valence.
- arousal-NRC-VAD-Lexicon-v2.1.txt: Includes arousal scores. The words are sorted in decreasing order of arousal.
- dominance-NRC-VAD-Lexicon-v2.1.txt: Includes dominance scores. The words are sorted in decreasing order of dominance.

3. The directory 'OneFilePerDimension/PolarSubset' has a version of the lexicon that only
includes polar terms -- terms with scores <= -0.333 or >= 0.333.  (Other thresholds to
determine polar words may also be used.)

4. The directory 'Unigrams' has a version of the lexicon that only includes unigrams
(single words).

5. The directory 'MWE' has a version of the lexicon that only includes multi-word
expressions.

PAPERS

P1. Paper-VAD-v2-2025.pdf: Research paper describing the NRC VAD Lexicon v2.

P2. Paper-VAD-ACL2018.pdf: Research paper describing the NRC VAD Lexicon v1.

P3. Paper-Practical-Ethical-Considerations-Lexicons.pdf: Research paper describing
practical and ethical considerations in the effective use of emotion and sentiment
lexicons.

P4. Paper-Ethics-Sheet-Emotion-Recognition.pdf: Research paper discussing ethical
considerations involved in automatic emotion recognition 'Ethics Sheet for Automatic
Emotion Recognition and Sentiment Analysis'.

VERSION 1.0

Download Version 1.0 of the lexicon from the project webpage. It also includes
translations in over 100 languages. 


III. RELEVANT RESEARCH PAPERS
-----------------------------

Details of the NRC VAD Lexicon are available in these papers:

- NRC VAD Lexicon v2: Norms for Valence, Arousal, and Dominance for over 55k English
Terms. Saif M. Mohammad. arXiv preprint arXiv:2503.23547. 2025.

- Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words.  Saif M.
Mohammad. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics,
Melbourne, Australia, July 2018.

Copies of the papers are included in this package.

If you use the lexicon in your work, then:

- Cite the papers:

		@article{vad-v2,
			title={{NRC VAD Lexicon v2: Norms for Valence, Arousal, and Dominance for over 55k English Terms}}, 
			author={Saif M. Mohammad},
			year={2025},
			journal={arXiv preprint arXiv:2503.23547},
			url={https://arxiv.org/abs/2503.23547} 
		}

		@inproceedings{vad-acl2018,
  			title={Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words},
 			author={Mohammad, Saif M.},
			booktitle={Proceedings of The Annual Conference of the Association for Computational Linguistics (ACL)},
			year={2018},
			address={Melbourne, Australia}
		}	

- Point to the lexicon homepage: 
		http://saifmohammad.com/WebPages/nrc-vad.html

Other relevant papers:

- Best Practices in the Creation and Use of Emotion Lexicons.  Saif M. Mohammad. Findings
  of the Association for Computational Linguistics: EACL 2023. Dubrovnik, Croatia. 2023.

- Ethics Sheet for Automatic Emotion Recognition and Sentiment Analysis.
  Saif M. Mohammad. Computational Linguistics. 48 (2): 239–278. June 2022.



IV. PYTHON CODE TO ANALYZE EMOTIONS IN TEXT
-------------------------------------------

There are many third party software packages that can be used in conjunction with the NRC
VAD Lexicon to analyze emotion word use in text. We recommend Emotion Dynamics:

	https://github.com/Priya22/EmotionDynamics

It is the primary package that we use to analyze text using the NRC Emotion Lexicon and
the NRC VAD Lexicon.  It can be used to generate a csv file with a number of emotion
features pertaining to the text of interest, including metrics of utterance emotion
dynamics.

See this paper for an example of the use of the lexicon to analyze emotions in text:

	Tweet Emotion Dynamics: Emotion Word Usage in Tweets from US and Canada. Krishnapriya
	Vishnubhotla and Saif M. Mohammad. In Proceedings of the 13th Language Resources and
	Evaluation Conference (LREC-2022), May 2022, Marseille, France.

	https://arxiv.org/pdf/2204.04862.pdf


V. NRC VAD LEXICON IN VARIOUS LANGUAGES
---------------------------------------

The NRC VAD Lexicon has annotations for English words. Despite some cultural differences,
it has been shown that a majority of affective norms are stable across languages. Thus, we
provide versions of the lexicon (version 1.0) in over 100 languages by translating the
English terms using Google Translate (August 2022).  [Download v1 of the lexicon from the
lexicon webpage for this.]

The lexicon is thus available for English and these languages:

Afrikaans, Albanian, Amharic, Arabic, Armenian, Azerbaijani, Basque, Belarusian, Bengali,
Bosnian, Bulgarian, Burmese, Catalan, Cebuano, Chichewa, Corsican, Croatian, Czech,
Danish, Dutch, Esperanto, Estonian, Filipino, Finnish, French, Frisian, Gaelic, Galician,
Georgian, German, Greek, Gujarati, HaitianCreole, Hausa, Hawaiian, Hebrew, Hindi, Hmong,
Hungarian, Icelandic, Igbo, Indonesian, Irish, Italian, Japanese, Javanese, Kannada,
Kazakh, Khmer, Kinyarwanda, Korean, Kurmanji, Kyrgyz, Lao, Latin, Latvian, Lithuanian,
Luxembourgish, Macedonian, Malagasy, Malay, Malayalam, Maltese, Maori, Marathi, Mongolian,
Nepali, Norwegian, Odia, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian,
Samoan, Sanskrit, Serbian, Sesotho, Shona, Simplified, Sindhi, Sinhala, Slovak, Slovenian,
Somali, Spanish, Sundanese, Swahili, Swedish, Tajik, Tamil, Tatar, Telugu, Thai,
Traditional, Turkish, Turkmen, Ukranian, Urdu, Uyghur, Uzbek, Vietnamese, Welsh, Xhosa,
Yiddish, Yoruba, Zulu

Note that an earlier version included translations obtained in 2018. The current 2022
translations are markedly better. That said, some of the translations are still incorrect
or they may simply be transliterations of the original English terms.


VI. USING POLAR TERMS ONLY, RESCALING, LEMMATIZATION, AND OTHER TECHNIQUES THAT MAY BE BENEFICIAL
-------------------------------------------------------------------------------------------------

The lexicon file can be used as is, but occasionally certain additional techniques can be
applied to make the most of it for one's specific application context.

1. POLAR SUBSET: For some applications it is more suitable to ignore neutralish words
(terms with V/A/D scores close to the middle of the scale) and only consider the more
polar words (terms with V/A/D scores close to the two ends of the scale).  There is no one
universally ``correct'' threshold; different thresholds simply make the polar and neutral
classes more or less restrictive. We provide one version of such a polar lexicon
(described below), but one can easily create their own version from the full lexicon by
excluding terms based on their own pre-chosen thresholds.  (One can also determine
threshold suitable for their application by tuning on a development set.)

The OneFilePerDimension/PolarSubset directory, has the file:
valence-polar-NRC-VAD-Lexicon-v2.0.txt which is a version of the valence lexicon with only
those entries that have scores less than or equal to -0.333 (negative words) and scores
greater than or equal to 0.333 (positive words).  Similarly, the directory has arousal and
dominance lexicons with only those entries that have scores less than or equal to -0.333
(low arousal/dominance words) and greater than or equal to 0.333 (high arousal/dominance
words). The entries with scores between -0.333 and 0.333 for a given dimension are
considered neutral for that dimension.

One can use other thresholds to be more permissive or restrictive for what is considered
polar (e.g., positive or negative).  For example, one can use the thresholds of -0.167 and
0.167 to be more permissive than the version provided in the package.

Disregarding the large number of neutralish words has an additional benefit of sharpening
the contrast when comparing average emotion scores across two sets of texts. For example,
when comparing the average valence across different genres of novels, including a large
number of neutralish terms in the lexicons leads to average scores close to the mid-point
score (0) for all genres, whereas only including polar terms will show greater
disparities in their average scores.


2. SCALE: The default form of the lexicon has entries with real-valued scores between -1
and 1.  However, the scores themselves have no inherent meaning other than being an
indication of relative score. (For example, a term with a higher valence score than
another term, is expected to be more positive than the other term.) Thus, for one's
specific application, if needed, the scores can be rescaled to other ranges such as 0 to
100 or 0 to 1.  Note that in version 1, the default form of the lexicon had scores from 0
to 1.

The default -1 to 1 scale version of the lexicon in v2 is of particular interest because:

a. Valence, arousal, and dominance are generally considered bipolar scales.  In the -1 to
1 scale, the highly polar terms have scores close to -1 and 1, and a large number of
neutralish terms have scores around 0. 

b. Highly polar words are often more informative for machine learning tasks than neutral
words.  Thus when features are derived from lexicons, it is often more useful for feature
values of polar terms to have higher absolute value scores than feature values of neutral
words.  Thus, using a -1 to 1 scale and taking the absolute values of the scores will
often lead to better features than deriving features using a 0 to 1 scale.


3. ANALYZING HIGH AND LOW V/A/D SEPARATELY: Analyzing high and low V/A/D  word usage
separately will provide more detailed insights than analyzing average V/A/D alone.  For
example, it helps distinguish cases with mostly neutral words from cases with many high-
and many low-valence words. It also helps determine whether greater use of high-V/A/D
words goes hand-in-hand with less frequent use of low-V/A/D word usage.


4. LEMMATIZATION: The lexicon largely includes the base forms or lemmas of words. For
example, it may include an entry for 'attack', but not for 'attacks' or 'attacking'. In
many cases, such morphological variants are expected to have similar emotion scores. So
one can first apply a third-party lemmatizer on the target text to determine the base
forms before applying the lexicon. Note that lemmatization must be applied with care; for
example, while it it good to go from 'helplessness' to 'helpless' (they are expected to
have similar emotional connotations), 'helplessness' should not be lemmatized to 'help'
(they are expected to have markedly different emotional connotations). Further, various
factors such as tense and 'differing predominant senses for different morphological forms'
can impact emotionality. So benefits of lemmatization are limited, especially when
analyzing large pieces of text.


5. OTHER: Other steps such as discarding highly ambiguous terms (terms with many
meanings), or identifying most common terms in one's text and inspecting emotion entries
in the lexicon for those terms (and correcting entries where appropriate), etc. are also
good practice.


VII. VERSION INFORMATION AND CHANGE LOG
---------------------------------------

- Version 2.1 is the latest version (Released March 2025).  
- Version 2.0 (Limited release October 2024).  
- Version 1.0  was Released in July 2018.  
- The automatic translations generated using Google Translate are updated every few years.  
  They were last obtained in August 2022. 
- The README was last updated in March 2025.


VIII. OTHER EMOTION LEXICONS
----------------------------

- The NRC Emotion Lexicon includes a list of more than 14,000 English words and their
associations with eight emotions (anger, fear, anticipation, trust, surprise, sadness,
joy, and disgust) and two sentiments (negative and positive). 
	
	Crowdsourcing a Word-Emotion Association Lexicon, Saif Mohammad and Peter Turney, Computational
    Intelligence, 29 (3), 436-465, 2013.

	http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm

- The NRC Emotion Intensity Lexicon is a list of English words (taken from the NRC Emotion
Lexicon and other sources) with real-valued scores of intensity for eight discrete emotions
(anger, anticipation, disgust, fear, joy, sadness, surprise, and trust).

	Word Affect Intensities. Saif M. Mohammad. In Proceedings of the 11th Edition of the Language 
	Resources and Evaluation Conference (LREC-2018), May 2018, Miyazaki, Japan.

    http://saifmohammad.com/WebPages/AffectIntensity.htm

- Words of Warmth (the Warmth, Sociability, Trust, and Competence Lexicons)

	[To be made available soon.]


- WorryWords (the Word--Anxiety Association Lexicon)
The NRC WorryWords Lexicon is a list of over 44,000 English words and real-valued scores
indicating their associations with anxiety: from -3 (maximum calmness) to 3 (maximum
anxiety). The scores were obtained by taking the means of individual labels by various
native speakers of English (from manual annotations through crowdsourcing).

	WorryWords: Norms of Anxiety Association for over 44k English Words. Saif M. Mohammad.
	In Proceedings of the Empirical Methods on Natural Language Processing (EMNLP 2024, Main),
	November 2024, Miami, FL.

	https://www.saifmohammad.com/worrywords.html


Various other emotion lexicons can be found here:
http://saifmohammad.com/WebPages/lexicons.html

You may also be interested in some of the other resources and work we have done on the
analysis of emotions in text:

http://saifmohammad.com/WebPages/ResearchAreas.html
http://saifmohammad.com/WebPages/ResearchInterests.html#EmotionAnalysis


IX. TERMS OF USE
----------------

1. Research Use: The lexicon mentioned in this page can be used freely for non-commercial
research and educational purposes.

2. Citation: Cite the papers associated with the lexicon in your research papers and
articles that make use of them.

3. Media Mentions: In news articles and online posts on work using the lexicon, cite the
lexicon. For example: "We make use of the <resource name>, created by <author(s)> at the
National Research Council Canada." We would appreciate a hyperlink to the lexicon home
page and an email to the contact author (saif.mohammad@nrc-cnrc.gc.ca).  (Authors and
homepage information provided at the top of the README.)

4. Credit: If you use the lexicon in a product or application, then acknowledge this in
the 'About' page and other relevant documentation of the application by stating the name
of the resource, the authors, and NRC. For example: "This application/product/tool makes
use of the <resource name>, created by <author(s)> at the National Research Council
Canada." We would appreciate a hyperlink to the lexicon home page and an email to the
contact author (saif.mohammad@nrc-cnrc.gc.ca).

5. No Redistribution: Do not redistribute the data. Direct interested parties to the
lexicon home page.  You may not rent or license the use of the lexicon nor otherwise
permit third parties to use it.

6. Proprietary Notice: You will ensure that any copyright notices, trademarks or other
proprietary right notices placed by NRC on the lexicon remains in evidence.

7. Title: All intellectual property rights in and to the lexicon shall remain the property
of NRC. All proprietary interests, rights, unencumbered titles, copyrights, or other
Intellectual Property Rights in the lexicon and all copies thereof remain at all times
with NRC.

8. Commercial License: If interested in commercial use of the lexicon, contact the author:
saif.mohammad@nrc-cnrc.gc.ca

9. Disclaimer: National Research Council Canada (NRC) disclaims any responsibility for the
use of the lexicon and does not provide technical support. NRC makes no representation and
gives no warranty of any kind with respect to the accuracy, usefulness, novelty,
validity, scope, or completeness of the lexicon and expressly disclaims any implied
warranty of merchantability or fitness for a particular purpose of the lexicon.  That
said, the contact listed above welcomes queries and clarifications.

10 Limitation of Liability: You will not make claims of any kind whatsoever upon or
against NRC or the creators of the lexicon, either on your own account or on behalf of any
third party, arising directly or indirectly out of your use of the lexicon. In no event
will NRC or the creators be liable on any theory of liability, whether in an action of
contract or strict liability (including negligence or otherwise), for any losses or
damages incurred by you, whether direct, indirect, incidental, special, exemplary or
consequential, including lost or anticipated profits, savings, interruption to business,
loss of business opportunities, loss of business information, the cost of recovering such
lost information, the cost of substitute intellectual property or any other pecuniary loss
arising from the use of, or the inability to use, the lexicon regardless of whether you
have advised NRC or NRC has advised you of the possibility of such damages.


We will be happy to hear from you. For example,:
- telling us what you are using the lexicon for
- providing feedback regarding the lexicon;
- if you are interested in having us analyze your data for sentiment, emotion, and other affectual information;
- if you are interested in a collaborative research project. We regularly collaborate with graduate students,
post-docs, faculty, and research professional from Computer Science, Psychology, Digital Humanities,
Linguistics, Social Science, etc.

Email: Dr. Saif M. Mohammad (saif.mohammad@nrc-cnrc.gc.ca, uvgotsaif@gmail.com)


X. ETHICAL CONSIDERATIONS
-------------------------

Please see the papers below (included with the download) for ethical
considerations involved in automatic emotion detection and the use of emotion
lexicons. (These also act as the Ethics and Data Statements for the lexicon.)

- Ethics Sheet for Automatic Emotion Recognition and Sentiment Analysis.
Saif M. Mohammad. Computational Linguistics. 48 (2): 239–278. June 2022.

@article{mohammad-2022-ethics-sheet,
    title = "Ethics Sheet for Automatic Emotion Recognition and Sentiment Analysis",
    author = "Mohammad, Saif M.",
    journal = "Computational Linguistics",
    volume = "48",
    number = "2",
    month = jun,
    year = "2022",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2022.cl-2.1/",
    doi = "10.1162/coli_a_00433",
    pages = "239--278"
}

- Best Practices in the Creation and Use of Emotion Lexicons. 
Saif M. Mohammad. Findings of the Association for Computational Linguistics: EACL 2023.

@inproceedings{mohammad-2023-best,
    title = "Best Practices in the Creation and Use of Emotion Lexicons",
    author = "Mohammad, Saif M.",
    editor = "Vlachos, Andreas  and
      Augenstein, Isabelle",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2023",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-eacl.136/",
    doi = "10.18653/v1/2023.findings-eacl.136",
    pages = "1825--1836"
}

- Practical and Ethical Considerations in the Effective use of Emotion and Sentiment Lexicons.
(This is an earlier version of the paper above.)
Saif M. Mohammad. arXiv preprint arXiv:2011.03492. December 2020.

Note that the labels for words are *associations* (and not denotations). As noted in the
paper above, they are limited by when the dataset was annotated, by the people that
annotated them, historical perceptions, and biases. (See bullets c through h in the
paper). It is especially worth noting that identity terms, such as those referring to
groups of people may be particularly prone to inappropriate biases. Further, marginalized
groups have historically faced more negative perceptions. Thus some terms that are
associated with marginalized groups may be marked as having associations with negative
emotions by the annotators. For example, group X marked as being associated with negative
emotions could imply that they have historically faced negative emotions or that some
people have negative associations with group X, or there is some other reason for the
negative association. The exact relationship is not listed in the lexicon. In order to
avoid misinterpretation and misuse, and as recommended generally in the ethical
considerations paper, we have removed entries for a small number of terms (about 200) that
are associated with identity groups or refer to slurs or taboo words. For the vast majority of sentiment and emotion
analysis requirements this removal will likely have no impact or will be beneficial.

The list of identity terms used is taken from this list developed in 2019 on an offensive
language project (abusive language is often directed at some identity groups):

https://github.com/hadarishav/Ruddit/blob/main/Dataset/identityterms_group.txt

The list of taboo words is take from:

The list of slur words is taken from:
https://itg.nls.uk/wiki/LGBTQIA%2B_Slurs_and_Slang

Only some of these terms occurred in our lexicon. This is not an exhaustive list; users
may choose to filter out additional terms for their particular task. We also welcome
requests for removal of additional terms from the list. Simply email us with the term or
terms and reason for removal.


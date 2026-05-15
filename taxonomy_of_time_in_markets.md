# A Taxonomy of "Time" in High-Frequency Market Microstructure

A short literature review documenting how the most-cited references in
high-frequency trading and market microstructure name the *wall-clock*
notion of time — and what they name it *against*. The motivating
question is internal to the `takens-rs` codebase: what should we call
the `Physical` clock variant once we decide the term is not optimal?
The conclusion is methodological rather than rhetorical, so the review
is intended to be useful as a glossary for the paper itself.

---

## 1. Why this matters

The `takens-rs` pipeline already exposes three bucketing clocks —
a fixed wall-clock window, a fixed snapshot count, and a posted-volume
threshold — under the internal names `Physical`, `Event`, and
`InfoTime`. The triple is conceptually coherent, but the term
`Physical` is *not* the dominant academic name for its referent in
the HFT canon. Renaming it touches code, on-disk artifact paths, and
the paper, so it is worth pinning the name to what reviewers will
recognise rather than to internal preference. This document records
what the canonical references actually say.

## 2. Methodology of the review

The review takes ten widely-cited references that establish or
codify the vocabulary used to discuss alternative time scales in
high-frequency markets:

- Mandelbrot & Taylor (1967) [1] — the original "trading time"
  subordinated-process paper, against which the "clock time" baseline
  is defined.
- Clark (1973) [2] — finite-variance subordinated process; commonly
  cited in the same lineage as Mandelbrot & Taylor.
- Engle & Russell (1998) [3] — autoregressive conditional duration
  (ACD); introduces the calendar-time / transaction-time contrast in
  modern econometric form.
- Ané & Geman (2000) [4] — "Order flow, transaction clock, and
  normality of asset returns"; popularises the "transaction clock"
  / "stochastic clock" terminology.
- Madhavan (2000) [5] — the canonical microstructure survey.
- Hasbrouck (2007) [6] — *Empirical Market Microstructure*; standard
  graduate textbook.
- Easley, López de Prado & O'Hara (2012a) [7] — "The Volume Clock";
  the single most-cited paper introducing the volume-clock framing.
- Easley, López de Prado & O'Hara (2012b) [8] — companion VPIN / flow
  toxicity paper.
- Dacorogna, Gençay, Müller, Olsen & Pictet (2001) [9] — *An
  Introduction to High-Frequency Finance*; the FX-microstructure
  reference that *does* use "physical time" as a term of art.
- López de Prado (2018) [10] — *Advances in Financial Machine
  Learning* (AFML); Chapter 2 on financial data structures, which
  treats time / tick / volume / dollar bars as a family.

For each reference, the relevant question is: what *word* does the
author use to name the variable that ticks at a constant rate against
real-world physical seconds, and what is it contrasted against?

## 3. Terms in active use across the literature

The vocabulary clusters into a small number of synonymous or
near-synonymous terms. Listed in approximate decreasing order of
prevalence in HFT-adjacent citations:

### 3.1 Clock time

Used by Mandelbrot & Taylor [1] ("price changes over fixed clock
time intervals"), and by Easley, López de Prado & O'Hara [7]
throughout the *Volume Clock* paper. The natural contrast in this
lineage is *clock time vs. volume time*; the asymmetry of the pairing
is rhetorical — the term "Volume Clock" only lands because the
implicit reference is *the* clock, i.e. the wall-clock. This is
arguably the closest to a primary term of art in the canon.

### 3.2 Calendar time

Used by Ané & Geman [4], Engle & Russell [3], Hasbrouck [6], and
much of the econometrics literature on irregularly-spaced transaction
data. Engle's follow-up survey work even contains a section literally
titled "Calendar Time Conversion". This is the standard term in
econometric writing aimed at a methodological audience.

### 3.3 Chronological (time / intervals / sampling)

López de Prado's house phrasing in *AFML* [10] and in the *Volume
Clock* paper [7]. He refers to "chronological intervals",
"chronological-based trading", and "chronological sampling" when
contrasting time bars against information-driven bars. This term
sits closest to the contemporary financial-ML vocabulary that the
`takens-rs` pipeline lives in.

### 3.4 Wall-clock time

Used in survey contexts and in Hasbrouck's textbook [6] as a more
literal synonym for calendar time. Appears with growing frequency in
the agent-based-modelling and HFT-engineering literature where the
distinction with internal exchange / engine time matters.

### 3.5 Physical time

Used in Dacorogna, Gençay, Müller, Olsen & Pictet [9] —
*An Introduction to High-Frequency Finance*, which is the seminal
Olsen-lineage FX-microstructure reference. The book contrasts
"physical time" with "business time" and "θ-time" / "intrinsic
time" in the context of FX seasonality. The term is well-pedigreed
within that lineage but is not the dominant naming in the
volume-bar / VPIN / AFML cluster that `takens-rs` aligns with.

### 3.6 Other contrasting clocks (for completeness)

The literature names the *alternative* clocks even more carefully
than it names the baseline:

- **Trading time / transaction time** — Mandelbrot & Taylor [1],
  Engle & Russell [3]. Closest to the `Event` clock in
  `takens-rs`.
- **Tick time** — Engle & Russell [3]; same referent as
  transaction time but emphasising the granularity of the underlying
  message stream.
- **Volume time / dollar time** — Easley, López de Prado &
  O'Hara [7][8], López de Prado [10]. The latter book extends
  the family to *imbalance bars* and *run bars* under the
  information-driven label.
- **Stochastic clock / transaction clock** — Ané & Geman [4],
  Clark [2]; emphasising the random nature of the time-change
  rather than its conditioning variable.
- **Intrinsic time / θ-time / business time** — Dacorogna et
  al. [9]; FX-seasonality terminology.

## 4. Per-reference terminology cheat sheet

| Reference | Wall-clock name used | Contrast term |
|---|---|---|
| Mandelbrot & Taylor 1967 [1] | clock time | trading time |
| Clark 1973 [2] | (clock time, implicit) | subordinated time |
| Engle & Russell 1998 [3] | calendar time | transaction time / tick time |
| Ané & Geman 2000 [4] | calendar time | transaction clock / stochastic clock |
| Madhavan 2000 [5] | calendar time / clock time | event time |
| Hasbrouck 2007 [6] | calendar time / wall-clock time | event time / transaction time |
| Easley, López de Prado & O'Hara 2012a [7] | clock time / chronological / calendar time | volume clock |
| Easley, López de Prado & O'Hara 2012b [8] | clock time / calendar time | volume time |
| Dacorogna et al. 2001 [9] | **physical time** | business time / θ-time / intrinsic time |
| López de Prado 2018 [10] | chronological intervals / time bars | tick / volume / dollar / imbalance bars |

The term **`physical time`** appears as a term of art *only* in the
Olsen-lineage reference [9]. Every other reference in this set
uses some combination of "clock time", "calendar time",
"chronological", or "wall-clock".

## 5. Where does "Unix" fit?

"Unix" — in the sense of *Unix-epoch nanoseconds encoded as a 64-bit
integer* — is universally understood as a *representation* of a
timestamp, not as a methodological time scale. It does not appear as
a term of art in any of the ten references above. A reviewer familiar
with the *Volume Clock* paper would read a phrase like "the Unix
clock" as either a typo, a property of the data layer, or as one
implementation detail bleeding into the methodology section.

The cleanest separation is to keep the *methodological* name from
§3 and reserve "Unix" for the encoding sentence in the data
schema:

> "We refer to this as the chronological clock; in the
>  implementation it is represented as Unix nanoseconds in
>  `int64`."

That sentence does the explanatory work once, leaves the
methodology vocabulary untouched, and signals that the encoding is
deliberate without making it load-bearing on the conceptual story.

## 6. Candidate replacement names ranked

If the rename proceeds, four candidates have non-trivial literature
support. Listed in order of best alignment with the audience
`takens-rs` is targeting (the AFML / Volume-Clock cluster):

### `Chronological`

Best match for López de Prado's own usage in *AFML* [10] and in
the *Volume Clock* paper [7]. Pairs cleanly as
`Chronological / Event / InfoTime` — the three clocks all read as
nouns of comparable register. Mild caveat: the word "chronological"
sometimes just means "in time order" in plain English, so the first
occurrence in the paper wants a one-line definition. Strong
recommendation overall.

### `Clock`

Most literal alignment with the *Volume Clock* framing [7],
where "clock time" is the named contrast for "volume clock". The
downside is purely Rust-side: `Clock::Clock` is a redundant variant
name. Mitigations: rename the enclosing enum to `ClockKind` or
`BucketingKind` (the latter generalises better if a fourth scheme
is ever added). The renaming has a small ripple but the resulting
naming is more faithful to the canonical citation.

### `Calendar`

The econometric-canonical choice — Ané & Geman [4], Engle &
Russell [3], Hasbrouck [6]. Reads conservatively and is
unambiguous. Minor downside: "calendar" carries a slight connotation
of daily-or-coarser granularity in everyday English, which is at
odds with the 60-second window the variant actually represents. Safe
choice for an econometrics-leaning audience.

### `WallClock`

Used by Hasbrouck [6] and in much of the engineering-flavoured
HFT and agent-based modelling literature. Most literal — pairs
symmetrically with `Event` and `InfoTime` as a noun. Slightly more
applied / less academic than the three above.

## 7. Net recommendation

Of the four, `Chronological` aligns best with the AFML reader
vocabulary that the `takens-rs` and `tarel` pipelines already speak.
`Clock` is the most faithful to the *Volume Clock* paper but
requires renaming the enclosing enum. `Calendar` and `WallClock`
are both defensible.

`Physical` is also defensible — it is not wrong, it is just
under-cited in the AFML / VPIN cluster and over-cited in the
Olsen / Dacorogna FX cluster. Keeping it is not a problem as long
as the paper introduces the term explicitly with a one-line nod to
Dacorogna et al. [9] so reviewers locate it.

`Unix` is the only choice that would consistently confuse a
NeurIPS reviewer familiar with the canonical references and is not
recommended as an external term.

---

## References (IEEE format)

[1] B. B. Mandelbrot and H. M. Taylor, "On the distribution of stock
price differences," *Operations Research*, vol. 15, no. 6, pp.
1057–1062, Nov.–Dec. 1967, doi: 10.1287/opre.15.6.1057.

[2] P. K. Clark, "A subordinated stochastic process model with finite
variance for speculative prices," *Econometrica*, vol. 41, no. 1,
pp. 135–155, Jan. 1973, doi: 10.2307/1913889.

[3] R. F. Engle and J. R. Russell, "Autoregressive conditional
duration: A new model for irregularly spaced transaction data,"
*Econometrica*, vol. 66, no. 5, pp. 1127–1162, Sep. 1998, doi:
10.2307/2999632.

[4] T. Ané and H. Geman, "Order flow, transaction clock, and
normality of asset returns," *The Journal of Finance*, vol. 55, no.
5, pp. 2259–2284, Oct. 2000, doi: 10.1111/0022-1082.00286.

[5] A. Madhavan, "Market microstructure: A survey," *Journal of
Financial Markets*, vol. 3, no. 3, pp. 205–258, Aug. 2000, doi:
10.1016/S1386-4181(00)00007-0.

[6] J. Hasbrouck, *Empirical Market Microstructure: The
Institutions, Economics, and Econometrics of Securities Trading*.
New York, NY, USA: Oxford Univ. Press, 2007.

[7] D. Easley, M. M. López de Prado, and M. O'Hara, "The volume
clock: Insights into the high-frequency paradigm," *The Journal of
Portfolio Management*, vol. 39, no. 1, pp. 19–29, Fall 2012, doi:
10.3905/jpm.2012.39.1.019.

[8] D. Easley, M. M. López de Prado, and M. O'Hara, "Flow toxicity
and liquidity in a high-frequency world," *The Review of Financial
Studies*, vol. 25, no. 5, pp. 1457–1493, May 2012, doi:
10.1093/rfs/hhs053.

[9] M. M. Dacorogna, R. Gençay, U. A. Müller, R. B. Olsen, and O. V.
Pictet, *An Introduction to High-Frequency Finance*. San Diego, CA,
USA: Academic Press, 2001.

[10] M. M. López de Prado, *Advances in Financial Machine Learning*.
Hoboken, NJ, USA: Wiley, 2018.

---

## BibTeX entries

```bibtex
@article{mandelbrot1967distribution,
  author  = {Mandelbrot, Benoit B. and Taylor, Howard M.},
  title   = {On the Distribution of Stock Price Differences},
  journal = {Operations Research},
  volume  = {15},
  number  = {6},
  pages   = {1057--1062},
  year    = {1967},
  month   = dec,
  doi     = {10.1287/opre.15.6.1057},
}

@article{clark1973subordinated,
  author  = {Clark, Peter K.},
  title   = {A Subordinated Stochastic Process Model with Finite Variance for Speculative Prices},
  journal = {Econometrica},
  volume  = {41},
  number  = {1},
  pages   = {135--155},
  year    = {1973},
  month   = jan,
  doi     = {10.2307/1913889},
}

@article{engle1998acd,
  author  = {Engle, Robert F. and Russell, Jeffrey R.},
  title   = {Autoregressive Conditional Duration: {A} New Model for Irregularly Spaced Transaction Data},
  journal = {Econometrica},
  volume  = {66},
  number  = {5},
  pages   = {1127--1162},
  year    = {1998},
  month   = sep,
  doi     = {10.2307/2999632},
}

@article{ane2000order,
  author  = {An{\'e}, Thierry and Geman, H{\'e}lyette},
  title   = {Order Flow, Transaction Clock, and Normality of Asset Returns},
  journal = {The Journal of Finance},
  volume  = {55},
  number  = {5},
  pages   = {2259--2284},
  year    = {2000},
  month   = oct,
  doi     = {10.1111/0022-1082.00286},
}

@article{madhavan2000market,
  author  = {Madhavan, Ananth},
  title   = {Market Microstructure: {A} Survey},
  journal = {Journal of Financial Markets},
  volume  = {3},
  number  = {3},
  pages   = {205--258},
  year    = {2000},
  month   = aug,
  doi     = {10.1016/S1386-4181(00)00007-0},
}

@book{hasbrouck2007empirical,
  author    = {Hasbrouck, Joel},
  title     = {Empirical Market Microstructure: The Institutions, Economics, and Econometrics of Securities Trading},
  publisher = {Oxford University Press},
  address   = {New York, NY, USA},
  year      = {2007},
}

@article{easley2012volume,
  author  = {Easley, David and L{\'o}pez de Prado, Marcos M. and O'Hara, Maureen},
  title   = {The Volume Clock: Insights into the High-Frequency Paradigm},
  journal = {The Journal of Portfolio Management},
  volume  = {39},
  number  = {1},
  pages   = {19--29},
  year    = {2012},
  doi     = {10.3905/jpm.2012.39.1.019},
}

@article{easley2012flow,
  author  = {Easley, David and L{\'o}pez de Prado, Marcos M. and O'Hara, Maureen},
  title   = {Flow Toxicity and Liquidity in a High-Frequency World},
  journal = {The Review of Financial Studies},
  volume  = {25},
  number  = {5},
  pages   = {1457--1493},
  year    = {2012},
  month   = may,
  doi     = {10.1093/rfs/hhs053},
}

@book{dacorogna2001highfrequency,
  author    = {Dacorogna, Michel M. and Gen{\c{c}}ay, Ramazan and M{\"u}ller, Ulrich A. and Olsen, Richard B. and Pictet, Olivier V.},
  title     = {An Introduction to High-Frequency Finance},
  publisher = {Academic Press},
  address   = {San Diego, CA, USA},
  year      = {2001},
}

@book{lopezdeprado2018afml,
  author    = {L{\'o}pez de Prado, Marcos M.},
  title     = {Advances in Financial Machine Learning},
  publisher = {Wiley},
  address   = {Hoboken, NJ, USA},
  year      = {2018},
}
```

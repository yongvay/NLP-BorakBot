# Part B — numbering and cross-reference fix list

Everything below follows from adding **2.0 Research Background** and **3.0 Methodology**.
Work through it in order. Steps 3 and 4 are Find & Replace in Google Docs with
**"Match using regular expressions" ticked**.

---

## 1. Section numbering

Target structure:

| New | Section | Was |
|---|---|---|
| 1.0 | Introduction | 1.0 |
| **2.0** | **Research Background** | *new* |
| **3.0** | **Methodology** | *new* |
| 4.0 | Pseudocode | 2.0 |
| 5.0 | Algorithm Analysis | 3.0 |
| 6.0 | Coding | 4.0 |
| 7.0 | Results | 5.0 |
| 8.0 | Discussion and Conclusion | 6.0 |
| 9.0 | References | 7.0 |

Renumber the **headings** in descending order so nothing collides: 7.x → 9.x, then
6.x → 8.x, 5.x → 7.x, 4.x → 6.x, 3.x → 5.x, 2.x → 4.x. Do the sub-headings with each
parent (e.g. 3.1–3.21 all become 5.1–5.21).

---

## 2. Figures

There are currently **two figures both numbered Figure 1**.

1. **Move the architecture diagram out of the Introduction into §3.1.** It stays *Figure 1*.
2. The training-loss curve in what is now §5.13: change its caption from *Figure 1* to
   **Figure 2**. Nothing in the body text refers to it by number, so this is the only edit.

---

## 3. Tables — 21 in total, currently out of document order

Tables 19 and 20 (in Results) currently sit *before* Tables 15–18 (in the Appendices).
The new §2.7 table also has to come first. Final order:

| Where | Caption now | Becomes |
|---|---|---|
| §2.7 Research gap | *new* | **Table 1** |
| §5.1 – §5.16 | Tables 1 – 13 | Tables 2 – 14 |
| §6.6 Repository structure | Table 14 | Table 15 |
| §7.1 Headline measurements | Table 19 | **Table 16** |
| §7.4 Objectives | Table 20 | **Table 17** |
| Appendix C | Table 15 | Table 18 |
| Appendix D | Table 16 | Table 19 |
| Appendix E | Table 17 | Table 20 |
| Appendix E | Table 18 | Table 21 |

**Do it in two passes, or numbers will overwrite each other.** Pass 1 renames everything
to a temporary `TBL` marker; pass 2 turns `TBL` back into `Table`. Regex on, one line at
a time, "Replace all":

**Pass 1**

```
\bTable 20\b   ->  TBL 17
\bTable 19\b   ->  TBL 16
\bTable 18\b   ->  TBL 21
\bTable 17\b   ->  TBL 20
\bTable 16\b   ->  TBL 19
\bTable 15\b   ->  TBL 18
\bTable 14\b   ->  TBL 15
\bTable 13\b   ->  TBL 14
\bTable 12\b   ->  TBL 13
\bTable 11\b   ->  TBL 12
\bTable 10\b   ->  TBL 11
\bTable 9\b    ->  TBL 10
\bTable 8\b    ->  TBL 9
\bTable 7\b    ->  TBL 8
\bTable 6\b    ->  TBL 7
\bTable 5\b    ->  TBL 6
\bTable 4\b    ->  TBL 5
\bTable 3\b    ->  TBL 4
\bTable 2\b    ->  TBL 3
\bTable 1\b    ->  TBL 2
```

**Pass 2**

```
TBL   ->  Table
```

This handles captions and in-text mentions together — the in-text references to
Tables 2, 5, 6, 11 and 13, and the five "Table N" cells inside the headline results
table, all move with them.

Then paste in the new §2.7 table, captioned **Table 1**.

---

## 4. Cross-references — 26 of them are now stale

The report refers to Algorithm Analysis subsections by their old flat numbers
("section 14"), which stopped being correct when they became 3.x, and will be wrong
again as 5.x. Fix all of them to the **§5.N** form.

Do these **first**, before the shorter ones, so `section 1` cannot match inside
`section 18`:

```
Sections 1 to 8       ->  §5.1 to §5.8
sections 1 to 8       ->  §5.1 to §5.8
Sections 9 to 21      ->  §5.9 to §5.21
sections 14 to 17     ->  §5.14 to §5.17
sections 14 and 16    ->  §5.14 and §5.16
```

Then the singles (counts are how many occurrences to expect):

```
section 18   ->  §5.18   (2)
section 16   ->  §5.16   (3)
Section 16   ->  §5.16   (1)
section 15   ->  §5.15   (1)
section 14   ->  §5.14   (5)
section 13   ->  §5.13   (1)
section 11   ->  §5.11   (2)
section 10   ->  §5.10   (2)
Section 10   ->  §5.10   (1)
Section 7    ->  §5.7    (1)
Section 5    ->  §5.5    (2)
```

And the short §-form references that came in with the Results and Discussion sections:

```
§1–8     ->  §5.1–5.8
§9       ->  §5.9
§12–14   ->  §5.12–5.14
§14      ->  §5.14
§16      ->  §5.16
§17      ->  §5.17
§19      ->  §5.19
§20      ->  §5.20
```

> **Do not touch `§4`.** Every occurrence of "proposal §4" points at the Part A
> objectives, not at this report.

---

## 5. Part A references — two read oddly

Both are in what is now §5.18 and §5.20:

```
Section §5.2 of the proposal   ->  Proposal §5.2
Section §5.7 of the proposal   ->  Proposal §5.7
```

Everything else already reads "Proposal §5.6", "Proposal §7.4" and so on, which keeps
Part A's numbering distinguishable from this report's new §5.x.

---

## 6. References — nine to add

The current list is 16 entries, all tools and metrics. The Research Background cites
related studies, which the list does not yet carry. Add these (the first eight are
already in Part A's reference list — copy them across so the two documents agree):

- Chen, H., Liu, X., Yin, D., & Tang, J. (2017). A survey on dialogue systems. *ACM SIGKDD Explorations Newsletter*, 19(2), 25–35.
- Khalip, K. I., Ku Khalif, K. M. N., Mohd Aziz, M. K. B., & Gegov, A. (2025). Sentiment analysis of noisy Malay text using a large language model. *Proceedings of the International Exchange and Innovation Conference on Engineering & Sciences*, 11, 1904–1909.
- Lim, H. T., Huspi, S. H., & Ibrahim, R. (2021). A conceptual framework for Malay-English mixed-language question answering system. *2021 International Congress of Advanced Technology and Engineering (ICOTEN)*.
- Shamsuddin, A. M., Juan, S. S., Chua, S., & Bramantoro, A. (2024). Semi-automatic sentiment identification for Malay-English code-switched data. *Journal of Advanced Research Design*, 123(1), 198–212.
- Sitaram, S., Chandu, K. R., Rallabandi, S. K., & Black, A. W. (2020). A survey of code-switched speech and language processing. arXiv:1904.00784.
- Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., & Hashimoto, T. B. (2023). *Stanford Alpaca: An instruction-following LLaMA model*. GitHub.
- Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., & Hajishirzi, H. (2023). Self-Instruct: Aligning language models with self-generated instructions. *Proceedings of ACL 2023*, 13484–13508.
- Winata, G. I., Cahyawijaya, S., Liu, Z., Lin, Z., Madotto, A., & Fung, P. (2021). Are multilingual models effective in code-switching? *Proceedings of the Fifth Workshop on Computational Approaches to Linguistic Code-Switching*, 142–153.

One more is cited in §2.3 but appears in neither document's list, so **check this entry
against the original before submitting**:

- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459–9474.

---

## 7. Last

Right-click the Table of Contents → **Update Field**. Do this after everything above,
or you will do it twice.

---

## On "Spelling, Grammar and Writing Mechanics"

This is rubric item 6, and it is **not a section to write** — it is how the whole report
is judged: professional writing, consistent structure, accurate grammar and formatting,
consistent referencing. Worth 2.5 marks. Nothing to insert; it is earned by the
proofread pass.

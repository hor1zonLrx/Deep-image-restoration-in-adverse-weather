# Contributing

Thanks for helping maintain this adverse-weather image restoration resource!

This repository is the official companion to the survey paper *Deep Image
Restoration in Adverse Weather: A Survey* (Neural Networks, 205 (2027) 109472).
All entries in the method, dataset, and benchmark tables are **traceable to the
survey** (its Tables 2–13 and the corresponding sections).

## What you can contribute

- New papers or methods (preferably with an explicit note on whether they are
  already covered in the survey or are community additions beyond it)
- Official project/code links
- Dataset links and corrections
- Benchmark results (always with the source and evaluation setting)
- Taxonomy corrections
- Broken-link fixes

## Suggested format for a new method

Please provide:

1. **Method**: short method name (as printed in the survey, if covered)
2. **Paper**: full title
3. **Task**: dehazing / deraining / desnowing / other adverse-weather / video / all-in-one / general-purpose all-in-one
4. **Category**: CNN / GAN / Transformer / Mamba / Diffusion / MoE / Prompt / Routing / etc.
5. **Learning paradigm**: SL / UL / SSL / WSL (when applicable)
6. **Venue + year**
7. **Paper URL**
8. **Official code URL** (if available)
9. **One-sentence summary**

## Scope rules

- Entries that appear in the survey must match the survey's own grouping,
  venue/year, and description. When in doubt, cite the survey section/table.
- If you suggest a paper **not** (yet) covered by the survey, mark it clearly as
  a community addition (e.g., `(beyond survey)`) so that the two groups of
  entries are never confused.
- Do not add duplicate entries.
- If a benchmark value comes from a paper, mention the source and evaluation
  setting in the PR description. Note that the survey's benchmark tables
  compile published results rather than re-produced numbers under a unified
  implementation (see Section 5.4.1 of the survey).

## Pull requests

- Keep entries concise and consistent with the existing tables.
- Prefer official paper/project/code links.
- Pull requests that only change data should not modify the survey PDF or the
  figures extracted from it.

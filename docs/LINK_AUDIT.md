# Link audit — 2026-09-13

## Scope and result

This change adds links to the existing bilingual method/dataset overviews and
general-purpose model list. It does not revise scientific claims, taxonomy,
venues, years, dataset sizes, or benchmark numbers.

- 195 display positions per language, corresponding to 174 distinct papers.
- All 174 papers have a Paper URL; 103 use arXiv abstract pages.
- 113 distinct papers have an author-provided Code/Project URL in the catalog.
  There are 107 Code/Project placements in the method lists; some additional
  repositories are cataloged for dataset papers.
- 25 of 40 dataset rows have an official Dataset entry point.
- Section 1.4 citation clusters, additional papers mentioned only in prose,
  loss/metric citations, and repeated benchmark-table names are not converted
  in this pass.

| Section | Entries | Paper | Code/Project or Dataset |
| --- | ---: | ---: | ---: |
| 1.1 Dehazing | 50 | 50 | 31 |
| 1.2 Deraining | 47 | 47 | 31 |
| 1.3 Desnowing | 11 | 11 | 5 |
| 1.5 Video | 3 | 3 | 3 |
| 2.1 One-to-One | 9 | 9 | 7 |
| 2.2 One-to-Many | 24 | 24 | 19 |
| 2.3 General-purpose models | 11 | 11 | 11 |
| 3.1 Dehazing datasets | 13 | 13 | 10 |
| 3.2 Deraining datasets | 16 | 16 | 8 |
| 3.3 Desnowing datasets | 6 | 6 | 4 |
| 3.4 Multi-weather datasets | 5 | 5 | 3 |

## Evidence and maintenance

[The catalog](../data/paper_links.json) maps exact README labels to the manuscript's
BibTeX keys. Full titles and authors were used to disambiguate acronyms.
The survey's local TeX/Bib sources were read for identity matching, but neither
the manuscript source nor its full bibliography is published by this change.

The catalog's `sources` record paper pages, publisher metadata and/or author
repositories used in review. `checked` records identity-review date; it is not a
promise of ongoing availability. The `year` field retains the survey's value,
including online-first versus issue-year differences.

Paper preference: verified arXiv abstract page, then official proceedings,
publisher/DOI, or author-hosted copy. Dataset links lead to official instructions,
not necessarily direct archives. Downloads, licenses, model execution and
reproduction of benchmark results were not tested. An accessible repository
does not establish completeness or reproducibility.

See [CONTRIBUTING](../CONTRIBUTING.md#maintaining-paper--code--dataset-links)
for the single-catalog bilingual update workflow and checks.

## Reachability

The final automated check covered 297 unique catalog Paper/Code/Dataset URLs:
288 responded successfully, 9 require manual review, and none returned 404/410.
The checker uses headers/limited response opening, not bulk PDF/data downloads.

Seven DOI/publisher endpoints returned HTTP 403; two academic dataset pages
returned TLS errors in the command-line client. These are not labeled dead.
The NYU Depth V2 and D-HAZY pages were separately readable through web browsing.
Publisher access may still depend on network, browser or subscription.

A machine-readable snapshot is provided in
[link_reachability.json](link_reachability.json). Re-run the optional network
checker for current results. It checks catalog targets, not every outbound link
in the README or every download referenced by a project.

## Resolved and unresolved resource cases

- **JSTASR:** the paper prints `weitingen83`; the verified author repository
  uses `weitingchen83`. The corrected repository has the paper citation,
  implementation and SRRS instructions.
- **CSUD:** the short paper-advertised URL returned 404. The author's
  `CSUD-Unsupervised-Deraining-CVPR2025` repository matches the authors,
  title and arXiv ID and contains training/testing instructions.
- **DW-GAN / Two-branch dehazing:** repository redirects were resolved to the
  current `liuh127/NTIRE-2021-Dehazing-*` names.
- **Defusion:** the author repository announces code but contained no
  implementation at review time; displayed as **Project**, not Code.
- **ACLNet, LMHaze and AMIRNet:** actual implementation files are present;
  older “coming soon” README text was not used to label these unreleased.
- **UHD-Processor / AIRFormer:** paper-advertised repositories
  `lyd-2022/UHD-processor` and `chdwyb/AIRFormer` returned 404.
  No substitute implementation is presented as official.
- **SLER:** the paper promises a future code/model release; no verified
  code repository is linked.
- **Rain800:** the original repository warns about deleted dataset files;
  its Dataset cell remains blank pending download verification.
- **Rain100L/H versus Rain200L/H:** the verified JORDER README explicitly
  lists Rain100 data. A shared paper identity alone was not treated as proof
  of all four dataset variants; dataset cells remain pending in this pass.

## Source discrepancies retained for author review

These are separate from hyperlink maintenance and have not been silently fixed:

- Some survey venue abbreviations say CVPR where the linked publication is a
  CVPR workshop (e.g. Cycle-Dehaze, CCDM, Jo & Sim, Yu et al., DW-GAN).
- LMHaze/MoE-Mamba's author repository identifies ACM Multimedia Asia 2024;
  the survey table retains its ACMMM 2024 string.
- DDMSNet's arXiv and journal title wording differs (“Geometric” versus
  “Depth” priors). The entry remains tied to its original citation key.
- CSUD is labeled SL in the survey table, while the original paper and author
  repository explicitly describe an unsupervised method.

## Pending links

A dash means no verified link was added, not a claim that no resource exists.

Dataset rows still pending: FRIDA (Tarel et al., 2010); FRIDA2 (Tarel et al., 2012); HazeCOCO (Zhu et al., 2019a); Rain12 (Li et al., 2016); Rain100L (Yang et al., 2017); Rain100H (Yang et al., 2017); Rain200L (Yang et al., 2017); Rain200H (Yang et al., 2017); Rain800 (Zhang et al., 2019); RainCityscapes (Hu et al., 2019); HQ-RAIN (Chen et al., 2025b); SnowCityscapes (Zhang et al., 2021); SnowKITTI2012 (Zhang et al., 2021); AIR40K (Gao et al., 2024b); HAC (Wan et al., 2025).

Method entries without a verified Code/Project URL:

- §1.1: Deep DCP (Golts et al., 2019) (`golts2019unsupervised`)
- §1.1: SSID (Li et al., 2019a) (`li2019semi`)
- §1.1: MSCNN-HE (Ren et al., 2020) (`ren2020single`)
- §1.1: CCDM (Zhang et al., 2020b) (`zhang2020color`)
- §1.1: Light-DehazeNet (Ullah et al., 2021) (`ullah2021light`)
- §1.1: Jo et al. (Jo & Sim, 2021) (`jo2021multi`)
- §1.1: DCNet (Chen et al., 2021d) (`chen2021dcnet`)
- §1.1: Yin et al. (Yin & Liu, 2025) (`yin2025driving`)
- §1.1: Dehaze-cGAN (Li et al., 2018b) (`li2018single`)
- §1.1: DDN (Yang et al., 2018) (`yang2018towards`)
- §1.1: DehazeGAN (Zhu et al., 2018) (`zhu2018dehazegan`)
- §1.1: HRGAN (Pang et al., 2018) (`pang2018visual`)
- §1.1: CDNet (Dudhane & Murala, 2019) (`dudhane2019cdnet`)
- §1.1: Cycle-Defog2Refog (Liu et al., 2020) (`liu2020end`)
- §1.1: Park et al. (Park et al., 2020) (`park2020fusion`)
- §1.1: TMS-GAN (Wang et al., 2021b) (`wang2021tms`)
- §1.1: ODCR (Wang et al., 2024b) (`wang2024odcr`)
- §1.1: UBRFC-Net (Sun et al., 2024a) (`sun2024unsupervised`)
- §1.1: DiffLI2D (Yang et al., 2024b) (`yang2024unleashing`)
- §1.2: DerainNet (Fu et al., 2017a) (`fu2017clearing`)
- §1.2: DAF-Net (Hu et al., 2019) (`hu2019depth`)
- §1.2: RaindropAttention (Quan et al., 2019) (`quan2019deep`)
- §1.2: LPNet (Fu et al., 2019) (`fu2019lightweight`)
- §1.2: MOSS (Huang et al., 2021) (`huang2021memory`)
- §1.2: UDRDR (Liu et al., 2021b) (`liu2021unpaired`)
- §1.2: UDGNet (Yu et al., 2021a) (`yu2021unsupervised`)
- §1.2: DEMore-Net (Wang et al., 2025c) (`wang2025rethinking`)
- §1.2: RR-GAN (Zhu et al., 2019b) (`zhu2019singe`)
- §1.2: UD-GAN (Jin et al., 2019) (`jin2019unsupervised`)
- §1.2: DerainCycleGAN (Wei et al., 2021) (`wei2021deraincyclegan`)
- §1.2: JRGR (Ye et al., 2021) (`ye2021closing`)
- §1.2: NLCL (Ye et al., 2022b) (`ye2022unsupervised`)
- §1.2: SmartAssign (Wang et al., 2023c) (`wang2023smartassign`)
- §1.2: FreqMamba (Zou et al., 2024) (`zou2024freqmamba`)
- §1.2: MS-DEMamba (Cheng et al., 2025) (`cheng2025unlocking`)
- §1.3: DesnowNet (Liu et al., 2018) (`liu2018desnownet`)
- §1.3: DDMSNet (Zhang et al., 2021) (`zhang2021deep`)
- §1.3: PEUNet (Guo et al., 2025) (`guo2025deep`)
- §1.3: Li et al. (2019d) (`li2019single2`)
- §1.3: DesnowGAN (Jaw et al., 2020) (`jaw2020desnowgan`)
- §1.3: SmartAssign (Wang et al., 2023c) (`wang2023smartassign`)
- §2.1: AIRFormer (Gao et al., 2024b) (`gao2023frequency`)
- §2.1: All-in-One (Li et al., 2020) (`li2020all`)
- §2.2: SLER-IR (Peng et al., 2026) (`shurui2026sler`)
- §2.2: Wang et al. (2025a) (`wang2025adapting`)
- §2.2: UHD-Processor (Liu et al., 2025) (`liu2025uhd`)
- §2.2: DTPM (Ye et al., 2024) (`ye2024learning`)
- §2.2: ResFlow (Qin et al., 2025) (`qin2025reversing`)

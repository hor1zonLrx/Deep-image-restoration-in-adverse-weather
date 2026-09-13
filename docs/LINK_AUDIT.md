# Link audit — 2026-09-13

## Scope

The survey remains the source of truth for grouping, descriptions, venue/year,
dataset sizes, and benchmark values. This update changes navigation only;
the Overleaf manuscript is untouched.

Per language, the README now contains:

- 208 paper-link placements covering 186 distinct papers: the existing 195
  overview/list positions, 12 citations in Section 1.4, and InvDSNet in prose.
- 114 Code/Project placements, including one explicitly labeled code mirror.
- Dataset entry points for 29 of the 40 dataset rows, including two labeled mirrors.

Repeated benchmark names, loss/metric citations, and remaining prose-only
references have not all been converted. A dash means the resource has not been
confirmed here, not that it does not exist.

## Checks performed

The original 174 paper targets were rechecked against available paper metadata,
proceedings, publisher records, and author pages. The 113 previously cataloged
repositories were checked through their READMEs; acronym-only matches and short
READMEs received separate review. Newly added links were matched to the survey's
bibliography using titles and authors.

A final request check of the 318 distinct Paper/Code/Project/Dataset targets
in the README returned 309 successful responses and nine access errors:
seven publisher HTTP 403 responses and two TLS errors (D-HAZY and SnowMaster).
No target returned HTTP 404 or 410. SnowMaster was independently accessible
through GitHub's API; D-HAZY was readable through web browsing. These observations
are a dated check, not a guarantee of future availability.

The two README files have identical resource-link order and targets. After
removing added link markup and columns, their original prose and table values
match the pre-link baseline (`d5406d8`), including all benchmark numbers.

No models were executed, and dataset archives, pretrained weights, licenses,
or benchmark reproducibility were not comprehensively tested. A reachable project
page is not proof that all of its downloads work.

## Additions and resource notes

- **Deep DCP:** added the [authors' Deep Energy repository](https://github.com/AlonaGolts/Deep_Energy),
  as cited in the [author-hosted paper](https://elad.cs.technion.ac.il/wp-content/uploads/2019/03/ImageDehazing-TIP.pdf).
- **DAF-Net / RainCityscapes:** added [code and dataset instructions](https://github.com/xw-hu/DAF-Net)
  linked from the [author's publications](https://xw-hu.github.io/publications/).
  Cityscapes downloads may require an account.
- **DerainNet / LPNet:** added the author's [DerainNet](https://xueyangfu.github.io/projects/tip2017.html)
  and [LPNet](https://xueyangfu.github.io/projects/LPNet.html) code-download pages.
- **FreqMamba / DEMore-Net:** added [FreqMamba code](https://github.com/aSleepyTree/FreqMamba)
  and [DEMore-Net code](https://github.com/yz-wang/DEMore-Net), matching the links in their
  [ACM](https://doi.org/10.1145/3664647.3680862) and
  [Neural Networks](https://www.sciencedirect.com/science/article/pii/S0893608025006197) records.
- **InvDSNet:** added the [paper](https://doi.org/10.1109/TCSVT.2022.3233655)
  and [implementation](https://github.com/csxhtan/InvDSNet) linked by the
  [corresponding author](https://csyhquan.github.io/category/c_publication.html).
- **HQ-RAIN:** added the [dataset repository](https://github.com/cschenxiang/HQ-RAIN).
  Its copied BibTeX contains an inconsistent arXiv identifier; the README Paper
  link retains the matching [2310.03535 paper](https://arxiv.org/abs/2310.03535).
- **JORDER / Rain100L / Rain100H:** the accessible
  [resource copy](https://github.com/ZhangXinNan/RainDetectionAndRemoval#2-download)
  is labeled `mirror`, since author ownership was not established.
- **JSTASR / CSUD:** retained the previously verified author-repository URL fixes.
  **Defusion** remains labeled `Project`, not released code.
- **UHD-Processor / AIRFormer / SLER:** no confirmed accessible implementation
  was added. Other unconfirmed code/dataset targets likewise remain `—`.

## Maintenance

Edit both READMEs directly, following [CONTRIBUTING](../CONTRIBUTING.md#maintaining-paper--code--dataset-links).
No synchronization script or generated link catalog is required. Dataset resource
pages are acceptable; there is no requirement for a separate per-variant manifest.

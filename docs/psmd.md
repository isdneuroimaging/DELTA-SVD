---
icon: lucide/signpost
search:
  exclude: true
hide:
  - toc
---

# PSMD is now DELTA-SVD

You were redirected here from **psmd-marker.com**. The PSMD pipeline has a successor: **DELTA-SVD**.

DELTA-SVD builds on the same skeleton-based approach as PSMD and still reports **PSMD** as one of its endpoints, alongside **MSMD** and the free-water metric **MSFW**. It is optimised for longitudinal processing, eliminates CSF partial-volume effects more effectively and includes a built-in quality-control report.

> [!WARNING]
> **Values from the two pipelines are not comparable.** PSMD values produced by DELTA-SVD are not equivalent to those from the original PSMD tool. Do not mix results from both pipelines in the same analysis; process all data for a given study with a single pipeline. See the [FAQ](faq.md#how-does-delta-svd-relate-to-the-original-psmd-pipeline) for details.

## Where to go

- **Starting a new study?** Use DELTA-SVD. Begin with the [Overview](overview.md) and [Installation](install.md).
- **Continuing an existing study, or reproducing published PSMD values?** The original PSMD tool remains available on GitHub: [isdneuroimaging/psmd](https://github.com/isdneuroimaging/psmd).

[Get started with DELTA-SVD](index.md){ .md-button .md-button--primary }
[Original PSMD tool](https://github.com/isdneuroimaging/psmd){ .md-button }

## Citation

**DELTA-SVD:** see the [DELTA-SVD citation](index.md#citation).

**PSMD:**

> Baykara E, Gesierich B, Adam R, et al. A novel imaging marker for small vessel disease based on skeletonization of white matter tracts and diffusion histograms. *Ann Neurol.* 2016;80(4):581–592. [doi:10.1002/ana.24758](https://doi.org/10.1002/ana.24758)

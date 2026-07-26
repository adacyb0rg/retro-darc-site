# Retro-DARC: Function-Preserving Residual-Memory Adapters for Pretrained Language and World Models

Ada Cyborg · July 2026 · project page, deployed on GitHub Pages.

Live: https://adacyb0rg.github.io/retro-darc-site/

## Contents

- `index.html` — the release page (self-contained; no external assets)
- `quickstart.html` — attach → verify identity → train → causal audit
- `Retro-DARC.pdf` — the full 29-page technical report
- `Retro-DARC_litepaper.pdf` — the 8-page conference-format short version
- `retro-darc-adapter-kit.zip` — depth_adapter.py + trained pythia-410m adapter
  checkpoints + quickstart guide (2 MB)
- `figures/` — the paper's vector figures
- `favicon.svg` · `og-card.png` (`gen_og_card.py`) · `404.html` — site chrome

License: CC BY 4.0 (see `LICENSE`).

## Provenance

This repo mirrors `docs/` of the research repository
(`github.com/adacyb0rg/retro-darc` — paper source, harness, raw per-run JSONs,
checkpoints; public at release). To update the site, rebuild there and copy
`docs/` over this repo's root, then push.

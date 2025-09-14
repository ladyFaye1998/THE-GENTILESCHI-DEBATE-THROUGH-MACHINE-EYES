# THE-GENTILESCHI-DEBATE-THROUGH-MACHINE-EYES
Reproducible ML for the Gentileschi corpus: artist classifier (Artemisia vs Orazio), gender classifier, strict data governance, calibrated reports.

Gentileschi ML Attribution — Reproducible Study
This repository hosts a reproducible art-historical + machine-learning study of the Gentileschi corpus. It implements two complementary model families: (1) an artist-specific classifier trained on uncontested works by Artemisia and Orazio Gentileschi, and (2) a cross-artist gender-attribution classifier to probe putative sex-of-artist signals. Models use EfficientNet-B0 (with ResNet-50 as an alternative), a domain-aware preprocessing pipeline, and strict dataset governance (label tiers, leakage and domain-shift controls, image standardisation). The project outputs per-work probabilities with calibration plots, confusion matrices, saliency/attribution maps, and narrative reports aligned to the attribution ladder (A, A?, W/A, W, F). Results are decision-support—transparent about uncertainty—intended to complement connoisseurship, technical analysis, and documentary evidence rather than replace them.

Highlights

Reproducible data build from Wikidata/Commons with quality filters

Artist classifier (Artemisia ↔ Orazio) and cross-artist gender classifier

Calibration-aware evaluation; clear uncertainty reporting

Saliency maps linking model attention to art-historical features

Git LFS-ready for images; .gitignore tuned for common stacks

Ethics & scope
Attributions are working hypotheses, not verdicts. Where labels are disputed, the repo preserves provenance notes and expresses model confidence with calibrated metrics.

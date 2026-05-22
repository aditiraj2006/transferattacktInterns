# Core Baseline Code

This folder contains a vanilla-only transfer-attack baseline for face verification.

Included baseline attacks:
- PGD
- MI-FGSM
- TI-FGSM
- SI-NI-FGSM
- MI-ADMIX-DI-TI

Not included:
- extra objective-level modifications from other project branches
- API-specific evaluation code paths

The intent is to give interns a clean starting point for implementing newer attacks in a separate manner and comparing them against the existing vanilla baselines on the provided subset.

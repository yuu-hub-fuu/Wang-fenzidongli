# Third-party notices

* `atlas_bench/metrics/{core,analyze,report}.py` are ports of
  `scripts/analyze_ensembles.py` and `scripts/print_analysis.py` from
  [AlphaFlow](https://github.com/bjing2016/alphaflow) (commit 0408d7c),
  MIT License, Copyright (c) 2024 Bowen Jing, Bonnie Berger, Tommi Jaakkola.
* `atlas_bench/data/splits/atlas_test.csv` is `splits/atlas_test.csv` from the
  same repository (MIT).
* `atlas_bench/tools/mdgen_prep_atlas.py` mirrors `scripts/prep_sims.py` of
  [MDGen](https://github.com/bjing2016/mdgen) (MIT).

The baselines themselves are not redistributed: `scripts/envs/*.sh` clone the
official repositories and download the authors' weights, which remain under
their own licenses (AlphaFlow MIT, ConfDiff Apache-2.0, BioEmu MIT, Str2Str
MIT, MDGen MIT, EBA MIT + Apache-2.0 (Protenix parts), BioKinema Apache-2.0 code /
CC BY-NC 4.0 data).
The ATLAS database (https://www.dsimb.inserm.fr/ATLAS) has its own terms of use.

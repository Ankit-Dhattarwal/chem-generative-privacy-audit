# Dataset Source and Licensing Information

## Source Benchmark
The molecular data used in this repository are derived from the **Molecular Sets (MOSES)** benchmark platform:
- **Paper**: Polykovskiy, Daniil, et al. "Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Models." *Frontiers in Pharmacology* 11 (2020): 565644. DOI: [10.3389/fphar.2020.565644](https://doi.org/10.3389/fphar.2020.565644)
- **Repository**: [https://github.com/molecularsets/moses](https://github.com/molecularsets/moses)
- **Download Source**: `https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv`

## License
The MOSES benchmark is distributed under the **MIT License**:
```text
MIT License

Copyright (c) 2018 Molecular Sets Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Dataset Files in this Directory
- `pool.csv`: 20,000 clean drug-like molecules sampled uniformly at random without replacement (`seed=0`), filtered to token length $\le 100$.
- `val.csv`: 500 disjoint validation molecules.
- `pool_metadata.csv`: Precomputed Murcko scaffolds, scaffold frequencies, and up to 10 randomized SMILES representations per molecule.
- `download_moses.py`: Standalone script to download the upstream MOSES dataset (`dataset_v1.csv`).
- `moses.csv`: (Optional / Excluded from release zip to save bandwidth) Full 81 MB upstream MOSES dataset table. Running `python data/download_moses.py` downloads this file if rebuilding the candidate pool from scratch. The provided `pool.csv`, `val.csv`, and `pool_metadata.csv` files are sufficient to run all experiments, evaluations, and table/figure generation directly.

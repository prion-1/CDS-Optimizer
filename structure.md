mRNA_optimizer/
├── main.ipynb
├── src/
│   ├── __init__.py
│   ├── gceh_module.py
│   ├── optimization.py
│   ├── pre_optimization.py
│   └── utils.py
├── data/
│   └── codon_tables/
│       ├── human.csv
│       ├── mouse.csv
│       ├── ecoli.csv
│       ├── scerevisiae.csv
│       └── spombe.csv
├── requirements.txt
└── README.md

=== requirements.txt ===

# Core dependencies
biopython>=1.79
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
ipywidgets>=8.0.0
scipy>=1.7.0
numba>=0.56.0
pytest>=7.0.0

# Optional dependencies
# jupyter>=1.0.0  # Only if running notebooks outside VS Code/GitHub/Colab
# Fallbacks in place in case no Vienna/numba
ViennaRNA>=2.5.0
numba
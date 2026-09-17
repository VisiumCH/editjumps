# Scope and Limitations

Key assumptions and constraints regarding model behavior:

- **No intrinsic variant scoring:** The editor generates variants without scoring quality or confidence. Distributional quality metrics require a natural sequence holdout from the target family. The model's internal rate field distinguishes corrupted sequences from natural sequences, but does not provide ranking signal among plausible variants.
- **Unconditional generation:** The editor operates without target property conditioning. It generates homologous sequence variants; experimental assays or downstream property models are required to assess functional fitness.
- **Approximate edit budgets:** The `--edits` parameter provides approximate guidance via clock calibration. Realized edit counts vary per sequence and are reported directly as Levenshtein distances in the output.

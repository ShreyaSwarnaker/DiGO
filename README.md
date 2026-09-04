# DiGO
DiGO is a sequence-based two-stage framework for protein function prediction. In Stage I, each protein becomes a directed graph over the twenty amino acid types, with an edge for every consecutive residue pair; separate source and target encoders preserve transition direction, and self-attention captures dependencies between non-adjacent residue types. This is fused through a learned gate with residue-level positional encodings and multi-layer ESM-2 embeddings, then classified under a flat multi-label objective with no ontological constraints. In Stage II the network is frozen and its outputs corrected without gradient updates: scores are propagated up the ontology to satisfy the true-path rule, temperature scaling corrects overconfidence, and per-term thresholds are fitted on validation data alone. Since the pipeline consumes only the sequence, it needs no structures, interaction networks, or homology search, and applies to every protein.

# Data
The primary evaluation uses the MSNGO benchmark. The original dataset was
introduced by the MSNGO authors and can be downloaded from 
[their repository](https://github.com/blingbell/MSNGO).
The exact CSV splits used in this work are included under `data/`, one folder
per ontology.

## Dataset-limited

A reconstruction of the `dataset-limited` benchmark from MSNGO (Wang et al., 2025),
restricted to the six experimental GO evidence codes - EXP, IDA, IPI, IMP, IGI,
IEP - excluding high-throughput and phylogenetically inferred annotations.

**Construction**: Built from GOA release 217 (2023-09-21), and `go-basic.obo` (GO release 2023-07-27).
Nine taxa were taken from the per-species GOA directories; the remaining four
(*Z. mays*, *S. pombe*, *E. coli*, *H. influenzae*) were extracted from
`goa_uniprot_all.gaf.217.gz`. Following MSNGO's procedure: retained the six evidence codes for the thirteen
target taxa; restrict to proteins present in `uniprot2string.txt`; drop
annotations dated on or after 2023-08-01; propagate labels to all ancestors under
`is_a` ∪ `part_of` and remove the ontology roots; split by minimum annotation date
at 2021-01-01 and 2022-08-01.


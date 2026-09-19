# What Was Read From Source and What Remains a Modelling Choice

The compression figures in this work depend on reimplementing the memory
accounting of Herrmann et al. correctly. This file separates what was verified
against their published code from what is still a judgement call, so a reader
can check the first and probe the second.

Reference implementation: `github.com/TinyAIoT/LightGBM-ToaD`, file
`src/treelearner/memory_restricted_forest.hpp`.

## Verified Against Source

**The bit-width helper.** Their `bits()` returns 1 for an input of zero and
`floor(log2(n)) + 1` otherwise. Reimplemented as `accounting.bits_width` and
checked in `tests/test_predictor.py`.

**The feature and threshold map.** Each used feature costs
`4 + bits(n_features - 1) + bits(max_thresholds_per_feature)` bits. The
constant 4 covers a 3-bit width code and a 1-bit numeric type flag.

**Threshold storage.** Boolean features cost 1 bit per threshold, everything
else 32 bits.

**Leaf storage.** Leaf values are held in a global table at 32 bits each, keyed
by a reserved sentinel feature identifier. That sentinel is feature id 255.

**Tree reference arrays.** Sized as `2^max_depth` leaf slots plus
`2^max_depth - 1` internal slots, keyed on the *configured* maximum depth
rather than the depth a tree actually reaches. This is the detail that drives
the depth result: a shallow tree inside a deep configuration still pays for the
entire array.

**Threshold indices.** Charged per real internal node at the width that node's
own feature needs, not as part of a fixed-width slot. This makes their layout
more efficient than a naive reading suggests, and it is why the succinct
alternative recovers less than the padding ratio alone would imply.

**Evaluated depths.** Their `runExperiments.sh` and `multiclassExperiments.sh`
sweep `for depth in 1 2 4 8` across regression, binary and multiclass. They do
not evaluate unbounded depth. The depth result in this paper is therefore a
scope condition on their layout rather than a defect in it, and it is written
that way.

## Modelling Choices

These are not in their code, because they concern the succinct layout proposed
here rather than the reference one. Each is varied in
`analysis.sensitivity_table`.

**Rank support overhead, default 25 percent.** A LOUDS bitmap needs auxiliary
structures for rank and select. Twenty-five percent is the usual budget in the
literature, but it is a budget rather than a measurement. Varying it from 0 to
100 percent moves the depth-eight figure between 2.34x and 2.53x, and the
crossover stays between depth four and depth five throughout.

**Float threshold width, default 32 bits.** Halving it to 16 bits leaves the
depth-eight figure at 2.53x.

**Decode cost is not modelled.** Rank operations add work at inference time.
Herrmann et al. already report their layout running several times slower per
prediction than a plain implementation, and nothing here extends that
measurement. The succinct layout would add to it.

## Known Gaps

Their penalised training arm was not reproduced. No fork of LightGBM was built,
so the comparison for that component is against their published figures rather
than a rerun. Five of their eight datasets were used.

Memory is counted, not measured on device. The numbers give stored size, not
flash occupancy after toolchain effects.

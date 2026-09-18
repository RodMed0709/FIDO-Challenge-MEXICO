# Results

| Task | Score | Components | Cases | Timeouts |
|---|---:|---|---:|---:|
| Task 1 | 0.625455 | keypoint AUC 0.854545; distance AUC 0.090909 | 5 | 0 |
| Task 2 | 0.000000 | corner AUC 0.000000 | 5 | 0 |

Task 1 improves the previous `r03` local score from `0.600000` to `0.625455`
on the same Mock Test harness. The gain comes from the new keypoint checkpoint;
the distance head is unchanged from `r03`.

Task 2 does not improve the Mock Test score. Its train-fold synthetic scale gate
passed, but that result did not transfer to this out-of-distribution set.

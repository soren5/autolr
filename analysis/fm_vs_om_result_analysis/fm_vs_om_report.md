# FM vs OM Result Analysis

## Inputs
- FM: `/Users/soren/Work/autolr/dumps/facilitated_mutation_base`
- OM: `/Users/soren/desktop_back_up/_Organized_Results/Original_AutoLR_experiments/adaptiveTest`

## Fitness Convention

- Raw evolutionary fitness was interpreted as `minimize`.
- Reported `quality` is higher-is-better. With the default framework convention, `quality = -fitness`.
- `budget_created_before` is the number of solutions logged in earlier generations of the same run. With constant population size, this is `generation_index * population_size`.
- This is a creation-budget axis, not an F-race evaluation-call axis.

## Comparability Notes

- Run counts differ by condition: {'FM': 5, 'OM': 10}.
- Observed final-generation ranges differ: {'FM': {'min': 199, 'max': 199}, 'OM': {'min': 453, 'max': 1500}}.
- Median final population sizes differ: {'FM': 100.0, 'OM': 20.0}.

## Directly Supported Findings

- The tables and plots below are directly computed from parsed `iteration_*.json` files without modifying the dumps.
- Threshold hits use final/run trajectory quality and the thresholds supplied to the script.

### Final Fitness Summary

| condition | runs | completed_runs_by_observed_condition_max | final_generation_min | final_generation_median | final_generation_max | final_population_size_median | final_budget_created_before_median | final_budget_created_before_max | final_best_quality_mean | final_best_quality_median | final_best_quality_max | final_best_quality_std | final_population_mean_quality_mean | final_population_median_quality_mean | runs_reaching_quality_0.6 | runs_reaching_quality_0.7 | runs_reaching_quality_0.8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FM | 5 | 5 | 199 | 199.0 | 199 | 100.0 | 19900.0 | 19900.0 | 0.8332945466041565 | 0.8357999920845032 | 0.8392000198364258 | 0.006894061207274597 | 0.6154910160824657 | 0.817629086971283 | 5 | 5 | 5 |
| OM | 10 | 8 | 453 | 1500.0 | 1500 | 20.0 | 30000.0 | 30000.0 | 0.8323714375495911 | 0.828000009059906 | 0.8648571372032166 | 0.01849583769516054 | 0.20679857262875884 | 0.1001428585499525 | 10 | 10 | 10 |

### Budget Comparison

| scope | condition | run_id | shared_budget_created_before | condition_max_budget_created_before | runs_total | runs_reaching_shared_budget | selected_generation | selected_budget_created_before | selected_best_quality | selected_best_quality_all_runs | selected_population_median_quality | final_generation | final_budget_created_before | final_best_quality | post_shared_budget_quality_gain | reached_shared_budget |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| condition_summary_at_shared_budget | FM |  | 19900 | 19900.0 | 5.0 | 5.0 |  | 19900.0 | 0.8332945466041565 | 0.8332945466041565 | 0.817629086971283 |  | 19900.0 | 0.8332945466041565 | 0.0 | True |
| condition_summary_at_shared_budget | OM |  | 19900 | 30000.0 | 10.0 | 8.0 |  | 19900.0 | 0.8145714327692986 | 0.8150571465492249 | 0.10000000149011612 |  | 30000.0 | 0.8362142965197563 | 0.021642863750457764 | False |

### Threshold Hits

| condition | threshold_quality | run_count | hit_count | hit_rate | median_generation_to_threshold | earliest_generation_to_threshold | latest_generation_to_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FM | 0.6 | 5 | 5 | 1.0 | 5.0 | 2 | 12 |
| FM | 0.7 | 5 | 5 | 1.0 | 5.0 | 2 | 23 |
| FM | 0.8 | 5 | 5 | 1.0 | 7.0 | 2 | 24 |
| OM | 0.6 | 10 | 10 | 1.0 | 4.0 | 0 | 38 |
| OM | 0.7 | 10 | 10 | 1.0 | 6.5 | 1 | 38 |
| OM | 0.8 | 10 | 10 | 1.0 | 105.0 | 7 | 739 |

### Phenotype Structure Summary

| condition | best_individuals | token_count_mean | token_count_median | char_length_mean | max_parenthesis_depth_mean | grad_reference_mean | state_reference_mean | constant_reference_mean | numeric_literal_mean | architecture_reference_mean | repeated_subexpression_proxy_mean | state_update_self_reference_proxy_rate | strides_presence_rate | kernel_size_presence_rate | filters_presence_rate | dilation_rate_presence_rate | padding_presence_rate | units_presence_rate | pool_size_presence_rate | layer_count_presence_rate | layer_num_presence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FM | 5 | 31.2 | 34.0 | 123.2 | 4.4 | 1.6 | 0.4 | 1.8 | 3.6 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| OM | 10 | 23.5 | 22.0 | 80.9 | 3.5 | 1.2 | 1.6 | 1.2 | 1.2 | 0.0 | 0.1 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### Early-To-Late Predictiveness

| condition | checkpoint_generation | paired_runs | pearson_correlation | spearman_correlation | checkpoint_quality_mean | final_quality_mean |
| --- | --- | --- | --- | --- | --- | --- |
| FM | 25 | 5 | 0.7834329411465293 | 0.8999999999999998 | 0.8271527290344238 | 0.8332945466041565 |
| OM | 25 | 10 | -0.1543563759224037 | -0.06686960980480709 | 0.596571435034275 | 0.8323714375495911 |
| FM | 50 | 5 | 0.8261611240851946 | 0.7999999999999999 | 0.8298618197441101 | 0.8332945466041565 |
| OM | 50 | 10 | -0.2712585781159427 | -0.07317073170731708 | 0.7566571474075318 | 0.8323714375495911 |
| FM | 100 | 5 | 0.9817450217526125 | 0.9999999999999999 | 0.8326072692871094 | 0.8332945466041565 |
| OM | 100 | 10 | 0.06213223153558594 | -0.34650615989763683 | 0.7181999951601028 | 0.8323714375495911 |
| OM | 200 | 10 | 0.40766142915545034 | 0.36474332620803873 | 0.811771422624588 | 0.8323714375495911 |

### F-Race Diagnostics

| condition | root | run_id | f_race_summary_files | selection_audit_summary_files | generations_logged | extra_evaluations | eliminated_count | tournament_events | tournament_changed | audit_unavailable_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FM | /Users/soren/Work/autolr/dumps/facilitated_mutation_base |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| OM | /Users/soren/desktop_back_up/_Organized_Results/Original_AutoLR_experiments/adaptiveTest |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Suggestive But Confounded Findings

- Differences between FM and OM should be treated as suggestive unless dataset/task sequence, grammar, and racing configuration are confirmed comparable.
- If one condition has a higher final budget, compare both the shared-budget table and the by-budget trajectory plots before interpreting final summaries.
- The phenotype structure features are proxies. They are useful for finding grammar/search symptoms such as bloat or architecture-variable usage, but they are not a semantic equivalence proof.

## Recommendation

Automated recommendation: At the shared budget of 19900 prior solutions, FM is not clearly worse than OM on mean selected best quality. OM's mean gain after the shared budget is 0.0216, which estimates the practical value of its larger search budget. OM also has the higher observed maximum quality. Because the comparison is confounded by observed setup differences, treat this as a prioritization signal rather than a causal result.

- Prefer current FM if its median trajectory and threshold hit rate are close to OM under comparable observed budgets.
- Prefer longer FM runs if the FM median best-quality trajectory is still climbing at the cutoff.
- Prefer an OM-like update-rule experiment if OM substantially outperforms FM and OM final phenotypes show more compact stateful structure.
- Prefer reducing architecture/noise first if architecture-variable usage is common in weak final best phenotypes.
- Prefer softer racing/thresholding if early-to-final correlations are weak and F-race logs show substantial early elimination.

## Final-Best Phenotype Examples

### FM run_1

- generation: 199
- quality: 0.8234363794326782
- archive key: `divide_no_nan(grad, constant(9.99916780e-01, dtype=float32))`

### FM run_2

- generation: 199
- quality: 0.8289818167686462
- archive key: `negative(multiply(add(constant(6.27124987e-04, dtype=float32), multiply(sqrt(constant(9.96851186e-01, dtype=float32)), grad)), constant(7.41067363e-02, dtype=float32)))`

### FM run_3

- generation: 199
- quality: 0.8392000198364258
- archive key: `divide_no_nan(negative(divide_no_nan(grad, subtract(grad, divide_no_nan(constant(9.76737464e-01, dtype=float32), constant(9.96851186e-01, dtype=float32))))), constant(9.99898151e-01, dtype=float32))`

### FM run_4

- generation: 199
- quality: 0.839054524898529
- archive key: `negative(multiply(constant(1.52235823e-01, dtype=float32), add(sigma, grad)))`

### FM run_5

- generation: 199
- quality: 0.8357999920845032
- archive key: `negative(multiply(constant(8.92170603e-02, dtype=float32), add(multiply(grad, negative(grad)), add(grad, beta))))`

### OM run_0

- generation: 1500
- quality: 0.8479999899864197
- archive key: `negative(multiply(add(beta, constant(1.27951705e-01)), add(beta, grad)))`

### OM run_1

- generation: 1500
- quality: 0.8471428751945496
- archive key: `multiply(add(alpha, constant(2.11963334e-01)), add(alpha, negative(grad)))`

### OM run_2

- generation: 1500
- quality: 0.8062857389450073
- archive key: `multiply(subtract(multiply(add(alpha, grad), constant(2.83228820e-02)), add(beta, grad)), constant(2.87185901e-01))`


## Artifacts

- `final_fitness_summary.csv`
- `budget_comparison.csv`
- `generation_summary.csv`
- `run_summary.csv`
- `threshold_hits.csv`
- `phenotype_structure_summary.csv`
- `early_late_predictiveness.csv`
- `f_race_diagnostics.csv`
- `individual_records_sample.csv` bounded by `--individual-record-sample`
- `full_individual_records.csv` only when `--write-full-individual-records` is used
- `best_fitness_trajectory.png`
- `best_fitness_by_budget.png`
- `population_fitness_trajectory.png`
- `population_fitness_by_budget.png`
- `checkpoint_vs_final_gen_<N>.png` when checkpoint data exists
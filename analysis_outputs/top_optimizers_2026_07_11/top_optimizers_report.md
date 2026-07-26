# Top Optimizers Report

Top N: `10`

Adjustment:
- non-multi adjusted score: `tiny_imagenet_score + 2.14013`
- multi adjusted score: sum of recorded per-task fitness scores
- task score total: sum of available recorded task fitness scores
- threshold source: `/Users/soren/desktop_back_up/_Organized_Results/cluster_dumps/2026-07-11/dumps/multi_arch_update/run_0/_parameters.json`

## Cross-Setup Adjusted Overall

- `multi_no_arch_update_phen_id_45499`
  - criterion: `adjusted_score` = `2.5155453979969025`
  - adjusted_score: `2.5155453979969025`
  - task_score_total: `2.5155453979969025`
  - overall_score: `1.7900012016296387`; fitness: `-1.7900012016296387`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_2` / `3324` / `143`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.7676666378974915, cifar100=0.6226666569709778, fmnist=0.8385454416275024, tiny_imagenet=0.2866666615009308`
  - operation/source: `mutation` / `iteration_143.json`
  - smart_phenotype: `multiply(negative(add(multiply(subtract(square(grad), beta), constant(2.11963334e-01)), grad)), sqrt(constant(1.90885420e-02)))`
  - advanced_readable_phenotype:

```text
beta = beta - negative(add(multiply(subtract(square(grad), beta), constant(2.11963334e-01)), grad))
weights = weights - multiply(beta, sqrt(constant(1.90885420e-02)))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: grad, lambda shape, alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(tf.math.square(grad), beta), tf.constant(2.11963334e-01, dtype=tf.float32)), grad)), lambda shape, alpha, beta, sigma, grad: tf.math.pow(tf.math.sqrt(tf.math.sqrt(grad)), grad), lambda shape, alpha, beta, sigma, grad: tf.math.multiply(beta, tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)))
```
- `multi_arch_update_phen_id_22661`
  - criterion: `adjusted_score` = `2.506127268075943`
  - adjusted_score: `2.506127268075943`
  - task_score_total: `2.506127268075943`
  - overall_score: `2.2934981763362883`; fitness: `-2.2934981763362883`
  - setup/run/genetic/generation: `multi_arch_update` / `run_4` / `1950` / `130`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `cifar10=0.7946666479110718, cifar100=0.6000000238418579, fmnist=0.8321272730827332, tiny_imagenet=0.27933332324028015`
  - operation/source: `mutation` / `iteration_130.json`
  - smart_phenotype: `negative(divide_no_nan(add(multiply(sqrt(constant(1.90885420e-02)), add(beta, add(add(square(grad), alpha), alpha))), grad), layer_count))`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(add(square(grad), alpha), alpha)
beta = beta - add(multiply(sqrt(constant(1.90885420e-02)), add(beta, alpha)), grad)
weights = weights - negative(divide_no_nan(beta, layer_count))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(tf.math.add(tf.math.square(grad), alpha), alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.multiply(tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)), tf.math.add(beta, alpha)), grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: grad, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(tf.math.divide_no_nan(beta, layer_count))
```
- `multi_no_arch_update_phen_id_42625`
  - criterion: `adjusted_score` = `2.501260608434677`
  - adjusted_score: `2.501260608434677`
  - task_score_total: `2.501260608434677`
  - overall_score: `2.7750133365392684`; fitness: `-2.7750133365392684`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_2` / `4809` / `235`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.7739999890327454, cifar100=0.6050000190734863, fmnist=0.8339272737503052, tiny_imagenet=0.28833332657814026`
  - operation/source: `mutation` / `iteration_235.json`
  - smart_phenotype: `multiply(negative(add(multiply(subtract(add(alpha, alpha), constant(2.11963334e-01)), beta), grad)), sqrt(constant(1.90885420e-02)))`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(alpha, alpha)
beta = beta - negative(add(multiply(subtract(alpha, constant(2.11963334e-01)), beta), grad))
weights = weights - multiply(beta, sqrt(constant(1.90885420e-02)))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: tf.math.add(alpha, alpha), lambda shape, alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(alpha, tf.constant(2.11963334e-01, dtype=tf.float32)), beta), grad)), lambda shape, alpha, beta, sigma, grad: tf.math.add(grad, sigma), lambda shape, alpha, beta, sigma, grad: tf.math.multiply(beta, tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)))
```
- `multi_no_arch_update_phen_id_42800`
  - criterion: `adjusted_score` = `2.500836342573166`
  - adjusted_score: `2.500836342573166`
  - task_score_total: `2.500836342573166`
  - overall_score: `3.262266665697098`; fitness: `-3.262266665697098`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_2` / `5392` / `282`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.7863333225250244, cifar100=0.612666666507721, fmnist=0.8358363509178162, tiny_imagenet=0.26600000262260437`
  - operation/source: `mutation` / `iteration_282.json`
  - smart_phenotype: `multiply(negative(add(multiply(subtract(alpha, constant(1.27951705e-01)), beta), grad)), sqrt(constant(1.90885420e-02)))`
  - advanced_readable_phenotype:

```text
alpha = alpha - alpha
beta = beta - negative(add(multiply(subtract(alpha, constant(1.27951705e-01)), beta), grad))
weights = weights - multiply(beta, sqrt(constant(1.90885420e-02)))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: alpha, lambda shape, alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(alpha, tf.constant(1.27951705e-01, dtype=tf.float32)), beta), grad)), lambda shape, alpha, beta, sigma, grad: tf.math.pow(tf.math.sqrt(tf.math.sqrt(tf.constant(2.11963334e-01, dtype=tf.float32))), tf.constant(8.59898661e-03, dtype=tf.float32)), lambda shape, alpha, beta, sigma, grad: tf.math.multiply(beta, tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)))
```
- `multi_arch_update_phen_id_22676`
  - criterion: `adjusted_score` = `2.4961575269699097`
  - adjusted_score: `2.4961575269699097`
  - task_score_total: `2.4961575269699097`
  - overall_score: `2.7786109030246733`; fitness: `-2.7786109030246733`
  - setup/run/genetic/generation: `multi_arch_update` / `run_4` / `1827` / `115`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `cifar10=0.8019999861717224, cifar100=0.5996666550636292, fmnist=0.8294909000396729, tiny_imagenet=0.26499998569488525`
  - operation/source: `mutation` / `iteration_115.json`
  - smart_phenotype: `negative(divide_no_nan(add(multiply(sqrt(constant(1.90885420e-02)), add(beta, add(square(grad), alpha))), grad), layer_count))`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(square(grad), alpha)
beta = beta - add(multiply(sqrt(constant(1.90885420e-02)), add(beta, alpha)), grad)
weights = weights - negative(divide_no_nan(beta, layer_count))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(tf.math.square(grad), alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.multiply(tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)), tf.math.add(beta, alpha)), grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: grad, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(tf.math.divide_no_nan(beta, layer_count))
```
- `multi_arch_update_phen_id_22729`
  - criterion: `adjusted_score` = `2.495327204465866`
  - adjusted_score: `2.495327204465866`
  - task_score_total: `2.495327204465866`
  - overall_score: `1.8081418097019195`; fitness: `-1.8081418097019195`
  - setup/run/genetic/generation: `multi_arch_update` / `run_4` / `1803` / `113`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `cifar10=0.7889999747276306, cifar100=0.590666651725769, fmnist=0.8313272595405579, tiny_imagenet=0.28433331847190857`
  - operation/source: `mutation` / `iteration_113.json`
  - smart_phenotype: `negative(divide_no_nan(add(multiply(sqrt(constant(1.90885420e-02)), add(grad, beta)), grad), layer_count))`
  - advanced_readable_phenotype:

```text
beta = beta - add(multiply(sqrt(constant(1.90885420e-02)), add(grad, beta)), grad)
weights = weights - negative(divide_no_nan(beta, layer_count))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(tf.math.square(filters), layer_count), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.multiply(tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)), tf.math.add(grad, beta)), grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(layer_num, layer_count), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(tf.math.divide_no_nan(beta, layer_count))
```
- `multi_no_arch_update_phen_id_44343`
  - criterion: `adjusted_score` = `2.492175668478012`
  - adjusted_score: `2.492175668478012`
  - task_score_total: `2.492175668478012`
  - overall_score: `3.275933337211609`; fitness: `-3.275933337211609`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_2` / `1232` / `28`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.7676666378974915, cifar100=0.6146666407585144, fmnist=0.8355090618133545, tiny_imagenet=0.27433332800865173`
  - operation/source: `mutation` / `iteration_28.json`
  - smart_phenotype: `multiply(negative(add(multiply(subtract(grad, beta), constant(2.11963334e-01)), grad)), sqrt(constant(1.90885420e-02)))`
  - advanced_readable_phenotype:

```text
beta = beta - negative(add(multiply(subtract(grad, beta), constant(2.11963334e-01)), grad))
weights = weights - multiply(beta, sqrt(constant(1.90885420e-02)))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: grad, lambda shape, alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(grad, beta), tf.constant(2.11963334e-01, dtype=tf.float32)), grad)), lambda shape, alpha, beta, sigma, grad: tf.math.pow(tf.math.sqrt(tf.math.sqrt(tf.constant(2.11963334e-01, dtype=tf.float32))), grad), lambda shape, alpha, beta, sigma, grad: tf.math.multiply(beta, tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)))
```
- `multi_arch_update_phen_id_22702`
  - criterion: `adjusted_score` = `2.4899697303771973`
  - adjusted_score: `2.4899697303771973`
  - task_score_total: `2.4899697303771973`
  - overall_score: `2.2934206068515777`; fitness: `-2.2934206068515777`
  - setup/run/genetic/generation: `multi_arch_update` / `run_4` / `1985` / `132`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `cifar10=0.7956666946411133, cifar100=0.5929999947547913, fmnist=0.8306363821029663, tiny_imagenet=0.2706666588783264`
  - operation/source: `mutation` / `iteration_132.json`
  - smart_phenotype: `negative(divide_no_nan(add(multiply(sqrt(constant(1.90885420e-02)), add(beta, divide_no_nan(square(grad), alpha))), grad), layer_count))`
  - advanced_readable_phenotype:

```text
alpha = alpha - divide_no_nan(square(grad), alpha)
beta = beta - add(multiply(sqrt(constant(1.90885420e-02)), add(beta, alpha)), grad)
weights = weights - negative(divide_no_nan(beta, layer_count))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.divide_no_nan(tf.math.square(grad), alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.multiply(tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)), tf.math.add(beta, alpha)), grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: grad, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(tf.math.divide_no_nan(beta, layer_count))
```
- `multi_no_arch_update_phen_id_44059`
  - criterion: `adjusted_score` = `2.4804848730564117`
  - adjusted_score: `2.4804848730564117`
  - task_score_total: `2.4804848730564117`
  - overall_score: `1.987210899591446`; fitness: `-1.987210899591446`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_2` / `3402` / `148`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.7739999890327454, cifar100=0.5993333458900452, fmnist=0.8328182101249695, tiny_imagenet=0.27433332800865173`
  - operation/source: `mutation` / `iteration_148.json`
  - smart_phenotype: `multiply(negative(add(multiply(subtract(grad, beta), add(constant(2.11963334e-01), constant(9.91401013e-01))), beta)), sqrt(constant(1.90885420e-02)))`
  - advanced_readable_phenotype:

```text
beta = beta - negative(add(multiply(subtract(grad, beta), add(constant(2.11963334e-01), constant(9.91401013e-01))), beta))
weights = weights - multiply(beta, sqrt(constant(1.90885420e-02)))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: grad, lambda shape, alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(grad, beta), tf.math.add(tf.constant(2.11963334e-01, dtype=tf.float32), tf.constant(9.91401013e-01, dtype=tf.float32))), beta)), lambda shape, alpha, beta, sigma, grad: tf.math.pow(tf.math.sqrt(tf.math.sqrt(tf.math.divide_no_nan(tf.constant(2.11963334e-01, dtype=tf.float32), grad))), tf.constant(8.59898661e-03, dtype=tf.float32)), lambda shape, alpha, beta, sigma, grad: tf.math.multiply(beta, tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)))
```
- `multi_no_arch_update_phen_id_45488`
  - criterion: `adjusted_score` = `2.479127287864685`
  - adjusted_score: `2.479127287864685`
  - task_score_total: `2.479127287864685`
  - overall_score: `2.7588888804117837`; fitness: `-2.7588888804117837`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_2` / `4432` / `209`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.7833333611488342, cifar100=0.593666672706604, fmnist=0.8371272683143616, tiny_imagenet=0.26499998569488525`
  - operation/source: `mutation` / `iteration_209.json`
  - smart_phenotype: `multiply(negative(add(multiply(subtract(square(grad), beta), constant(1.07052146e-01)), grad)), sqrt(constant(1.90885420e-02)))`
  - advanced_readable_phenotype:

```text
beta = beta - negative(add(multiply(subtract(square(grad), beta), constant(1.07052146e-01)), grad))
weights = weights - multiply(beta, sqrt(constant(1.90885420e-02)))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: tf.math.add(tf.math.divide_no_nan(grad, alpha), alpha), lambda shape, alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(tf.math.square(grad), beta), tf.constant(1.07052146e-01, dtype=tf.float32)), grad)), lambda shape, alpha, beta, sigma, grad: tf.math.pow(tf.math.sqrt(tf.math.sqrt(tf.math.add(alpha, grad))), tf.constant(2.11963334e-01, dtype=tf.float32)), lambda shape, alpha, beta, sigma, grad: tf.math.multiply(beta, tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)))
```

## Cross-Setup Task: cifar10

- `multi_arch_update_phen_id_22676`
- `multi_arch_update_phen_id_22713`
  - criterion: `cifar10_score` = `0.7963333129882812`
  - adjusted_score: `2.466387838125229`
  - task_score_total: `2.466387838125229`
  - overall_score: `1.3104994118213653`; fitness: `-1.3104994118213653`
  - setup/run/genetic/generation: `multi_arch_update` / `run_4` / `1923` / `127`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `cifar10=0.7963333129882812, cifar100=0.5706666707992554, fmnist=0.830054521560669, tiny_imagenet=0.2693333327770233`
  - operation/source: `mutation` / `iteration_127.json`
  - smart_phenotype: `negative(divide_no_nan(add(multiply(sqrt(constant(1.90885420e-02)), add(beta, multiply(square(grad), alpha))), grad), layer_count))`
  - advanced_readable_phenotype:

```text
alpha = alpha - multiply(square(grad), alpha)
beta = beta - add(multiply(sqrt(constant(1.90885420e-02)), add(beta, alpha)), grad)
weights = weights - negative(divide_no_nan(beta, layer_count))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.multiply(tf.math.square(grad), alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.multiply(tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)), tf.math.add(beta, alpha)), grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: grad, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(tf.math.divide_no_nan(beta, layer_count))
```
- `multi_arch_update_phen_id_22702`
- `multi_arch_update_phen_id_22661`
- `multi_arch_update_phen_id_22729`
- `multi_arch_update_phen_id_22716`
  - criterion: `cifar10_score` = `0.7889999747276306`
  - adjusted_score: `2.456624209880829`
  - task_score_total: `2.456624209880829`
  - overall_score: `1.7995866596698762`; fitness: `-1.7995866596698762`
  - setup/run/genetic/generation: `multi_arch_update` / `run_4` / `1824` / `115`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `cifar10=0.7889999747276306, cifar100=0.5916666388511658, fmnist=0.8332909345626831, tiny_imagenet=0.24266666173934937`
  - operation/source: `mutation` / `iteration_115.json`
  - smart_phenotype: `negative(divide_no_nan(add(multiply(sqrt(constant(1.90885420e-02)), add(beta, pool_size)), grad), layer_count))`
  - advanced_readable_phenotype:

```text
beta = beta - add(multiply(sqrt(constant(1.90885420e-02)), add(beta, pool_size)), grad)
weights = weights - negative(divide_no_nan(beta, layer_count))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(tf.math.square(alpha), grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.multiply(tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)), tf.math.add(beta, pool_size)), grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(tf.math.add(layer_count, layer_count), sigma), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(tf.math.divide_no_nan(beta, layer_count))
```
- `multi_no_arch_update_phen_id_42800`
- `multi_no_arch_update_phen_id_43849`
  - criterion: `cifar10_score` = `0.7856666445732117`
  - adjusted_score: `2.435806065797806`
  - task_score_total: `2.435806065797806`
  - overall_score: `2.457413339614868`; fitness: `-2.457413339614868`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_2` / `1213` / `27`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.7856666445732117, cifar100=0.5720000267028809, fmnist=0.8294727206230164, tiny_imagenet=0.2486666738986969`
  - operation/source: `mutation` / `iteration_27.json`
  - smart_phenotype: `multiply(negative(add(multiply(subtract(grad, add(constant(8.92170603e-02), grad)), grad), subtract(grad, multiply(constant(6.13831074e-02), beta)))), sqrt(constant(1.90885420e-02)))`
  - advanced_readable_phenotype:

```text
beta = beta - negative(add(multiply(subtract(grad, add(constant(8.92170603e-02), grad)), grad), subtract(grad, multiply(constant(6.13831074e-02), beta))))
weights = weights - multiply(beta, sqrt(constant(1.90885420e-02)))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: tf.constant(9.99916780e-01, dtype=tf.float32), lambda shape, alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(grad, tf.math.add(tf.constant(8.92170603e-02, dtype=tf.float32), grad)), grad), tf.math.subtract(grad, tf.math.multiply(tf.constant(6.13831074e-02, dtype=tf.float32), beta)))), lambda shape, alpha, beta, sigma, grad: tf.math.pow(grad, tf.math.sqrt(alpha)), lambda shape, alpha, beta, sigma, grad: tf.math.multiply(beta, tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)))
```
- `multi_no_arch_update_phen_id_45488`
- `multi_no_arch_update_phen_id_44390`
  - criterion: `cifar10_score` = `0.7789999842643738`
  - adjusted_score: `2.4742908477783203`
  - task_score_total: `2.4742908477783203`
  - overall_score: `1.5498444437980652`; fitness: `-1.5498444437980652`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_2` / `3067` / `123`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.7789999842643738, cifar100=0.5956666469573975, fmnist=0.8362908959388733, tiny_imagenet=0.2633333206176758`
  - operation/source: `mutation` / `iteration_123.json`
  - smart_phenotype: `multiply(negative(add(multiply(subtract(grad, beta), constant(2.87185901e-01)), add(grad, grad))), sqrt(constant(1.90885420e-02)))`
  - advanced_readable_phenotype:

```text
beta = beta - negative(add(multiply(subtract(grad, beta), constant(2.87185901e-01)), add(grad, grad)))
weights = weights - multiply(beta, sqrt(constant(1.90885420e-02)))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: grad, lambda shape, alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(grad, beta), tf.constant(2.87185901e-01, dtype=tf.float32)), tf.math.add(grad, grad))), lambda shape, alpha, beta, sigma, grad: tf.math.pow(tf.math.sqrt(tf.math.sqrt(tf.constant(2.11963334e-01, dtype=tf.float32))), tf.constant(8.59898661e-03, dtype=tf.float32)), lambda shape, alpha, beta, sigma, grad: tf.math.multiply(beta, tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)))
```

## Cross-Setup Task: cifar100

- `multi_no_arch_update_phen_id_45499`
- `multi_no_arch_update_phen_id_44343`
- `multi_no_arch_update_phen_id_42800`
- `multi_no_arch_update_phen_id_44144`
  - criterion: `cifar100_score` = `0.609333336353302`
  - adjusted_score: `2.4220727384090424`
  - task_score_total: `2.4220727384090424`
  - overall_score: `1.614626278479894`; fitness: `-1.614626278479894`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_2` / `2962` / `117`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.7753333449363708, cifar100=0.609333336353302, fmnist=0.8330727219581604, tiny_imagenet=0.2043333351612091`
  - operation/source: `mutation` / `iteration_117.json`
  - smart_phenotype: `multiply(negative(add(multiply(subtract(grad, beta), add(grad, constant(1.80176593e-01))), grad)), sqrt(constant(1.90885420e-02)))`
  - advanced_readable_phenotype:

```text
beta = beta - negative(add(multiply(subtract(grad, beta), add(grad, constant(1.80176593e-01))), grad))
weights = weights - multiply(beta, sqrt(constant(1.90885420e-02)))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: tf.constant(5.12469082e-04, dtype=tf.float32), lambda shape, alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(grad, beta), tf.math.add(grad, tf.constant(1.80176593e-01, dtype=tf.float32))), grad)), lambda shape, alpha, beta, sigma, grad: tf.math.add(tf.constant(2.11963334e-01, dtype=tf.float32), sigma), lambda shape, alpha, beta, sigma, grad: tf.math.multiply(beta, tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)))
```
- `multi_no_arch_update_phen_id_42625`
- `multi_no_arch_update_phen_id_44220`
  - criterion: `cifar100_score` = `0.6046666502952576`
  - adjusted_score: `2.4762302935123444`
  - task_score_total: `2.4762302935123444`
  - overall_score: `2.4814993917942045`; fitness: `-2.4814993917942045`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_2` / `2445` / `82`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.7713333368301392, cifar100=0.6046666502952576, fmnist=0.8335636258125305, tiny_imagenet=0.2666666805744171`
  - operation/source: `mutation` / `iteration_82.json`
  - smart_phenotype: `multiply(negative(add(multiply(subtract(grad, beta), constant(1.52235823e-01)), grad)), sqrt(constant(1.90885420e-02)))`
  - advanced_readable_phenotype:

```text
beta = beta - negative(add(multiply(subtract(grad, beta), constant(1.52235823e-01)), grad))
weights = weights - multiply(beta, sqrt(constant(1.90885420e-02)))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: grad, lambda shape, alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(grad, beta), tf.constant(1.52235823e-01, dtype=tf.float32)), grad)), lambda shape, alpha, beta, sigma, grad: tf.math.pow(tf.math.sqrt(tf.math.sqrt(tf.math.negative(tf.constant(2.11963334e-01, dtype=tf.float32)))), tf.math.add(tf.math.subtract(tf.constant(8.59898661e-03, dtype=tf.float32), tf.math.divide_no_nan(tf.math.divide_no_nan(grad, tf.math.negative(beta)), tf.math.subtract(alpha, tf.math.divide_no_nan(grad, grad)))), tf.math.square(tf.math.add(tf.math.add(tf.constant(9.92962885e-01, dtype=tf.float32), alpha), beta)))), lambda shape, alpha, beta, sigma, grad: tf.math.multiply(beta, tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)))
```
- `multi_no_arch_update_phen_id_44258`
  - criterion: `cifar100_score` = `0.6029999852180481`
  - adjusted_score: `2.4787999987602234`
  - task_score_total: `2.4787999987602234`
  - overall_score: `2.2875878691673277`; fitness: `-2.2875878691673277`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_2` / `2506` / `86`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.7776666879653931, cifar100=0.6029999852180481, fmnist=0.8348000049591064, tiny_imagenet=0.2633333206176758`
  - operation/source: `mutation` / `iteration_86.json`
  - smart_phenotype: `multiply(negative(add(multiply(subtract(grad, beta), constant(2.11963334e-01)), add(divide_no_nan(grad, alpha), grad))), sqrt(constant(1.90885420e-02)))`
  - advanced_readable_phenotype:

```text
alpha = alpha - divide_no_nan(grad, alpha)
beta = beta - negative(add(multiply(subtract(grad, beta), constant(2.11963334e-01)), add(alpha, grad)))
weights = weights - multiply(beta, sqrt(constant(1.90885420e-02)))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: tf.math.divide_no_nan(grad, alpha), lambda shape, alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(grad, beta), tf.constant(2.11963334e-01, dtype=tf.float32)), tf.math.add(alpha, grad))), lambda shape, alpha, beta, sigma, grad: tf.math.pow(tf.math.sqrt(tf.math.sqrt(tf.math.negative(grad))), tf.math.add(tf.math.subtract(grad, tf.math.divide_no_nan(tf.math.divide_no_nan(grad, tf.math.negative(beta)), tf.math.subtract(tf.constant(2.11963334e-01, dtype=tf.float32), tf.math.divide_no_nan(grad, grad)))), tf.math.square(tf.math.add(tf.math.add(beta, alpha), tf.constant(8.59898661e-03, dtype=tf.float32))))), lambda shape, alpha, beta, sigma, grad: tf.math.multiply(beta, tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)))
```
- `multi_no_arch_update_phen_id_44078`
  - criterion: `cifar100_score` = `0.6013333201408386`
  - adjusted_score: `2.3958484679460526`
  - task_score_total: `2.3958484679460526`
  - overall_score: `1.9215313245852788`; fitness: `-1.9215313245852788`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_2` / `2854` / `110`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.7710000276565552, cifar100=0.6013333201408386, fmnist=0.8291817903518677, tiny_imagenet=0.19433332979679108`
  - operation/source: `mutation` / `iteration_110.json`
  - smart_phenotype: `multiply(negative(add(multiply(subtract(grad, beta), add(constant(2.47663801e-01), grad)), grad)), sqrt(constant(1.90885420e-02)))`
  - advanced_readable_phenotype:

```text
beta = beta - negative(add(multiply(subtract(grad, beta), add(constant(2.47663801e-01), grad)), grad))
weights = weights - multiply(beta, sqrt(constant(1.90885420e-02)))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: grad, lambda shape, alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(grad, beta), tf.math.add(tf.constant(2.47663801e-01, dtype=tf.float32), grad)), grad)), lambda shape, alpha, beta, sigma, grad: tf.constant(2.11963334e-01, dtype=tf.float32), lambda shape, alpha, beta, sigma, grad: tf.math.multiply(beta, tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)))
```
- `multi_arch_update_phen_id_22661`
- `multi_arch_update_phen_id_22676`

## Cross-Setup Task: fmnist

- `multi_no_arch_update_phen_id_35199`
  - criterion: `fmnist_score` = `0.8469818234443665`
  - adjusted_score: `1.2903151512145996`
  - task_score_total: `1.2903151512145996`
  - overall_score: `1.0318262577056885`; fitness: `-1.0318262577056885`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_3` / `6253` / `195`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.44333332777023315, fmnist=0.8469818234443665`
  - operation/source: `mutation` / `iteration_195.json`
  - smart_phenotype: `multiply(add(subtract(alpha, add(multiply(grad, negative(pow(sqrt(pow(grad, alpha)), beta))), beta)), add(multiply(grad, negative(pow(sqrt(pow(grad, alpha)), beta))), beta)), constant(5.07243606e-02))`
  - advanced_readable_phenotype:

```text
alpha = alpha - alpha
beta = beta - add(multiply(grad, negative(pow(sqrt(pow(grad, alpha)), beta))), beta)
sigma = sigma - beta
weights = weights - multiply(add(subtract(alpha, sigma), beta), constant(5.07243606e-02))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: alpha, lambda shape, alpha, beta, grad: tf.math.add(tf.math.multiply(grad, tf.math.negative(tf.math.pow(tf.math.sqrt(tf.math.pow(grad, alpha)), beta))), beta), lambda shape, alpha, beta, sigma, grad: beta, lambda shape, alpha, beta, sigma, grad: tf.math.multiply(tf.math.add(tf.math.subtract(alpha, sigma), beta), tf.constant(5.07243606e-02, dtype=tf.float32))
```
- `multi_no_arch_update_phen_id_34826`
  - criterion: `fmnist_score` = `0.8447454571723938`
  - adjusted_score: `1.312745451927185`
  - task_score_total: `1.312745451927185`
  - overall_score: `1.3448121070861816`; fitness: `-1.3448121070861816`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_3` / `5806` / `187`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.46799999475479126, fmnist=0.8447454571723938`
  - operation/source: `mutation` / `iteration_187.json`
  - smart_phenotype: `multiply(add(subtract(add(multiply(negative(grad), pow(sqrt(square(add(alpha, square(constant(9.94242714e-01))))), square(grad))), beta), add(multiply(negative(grad), pow(sqrt(square(add(alpha, square(constant(9.94242714e-01))))), square(grad))), beta)), add(multiply(negative(grad), pow(sqrt(square(add(alpha, square(constant(9.94242714e-01))))), square(grad))), beta)), constant(5.07243606e-02))`
  - advanced_readable_phenotype:

```text
alpha = alpha - alpha
beta = beta - add(multiply(negative(grad), pow(sqrt(square(add(alpha, square(constant(9.94242714e-01))))), square(grad))), beta)
sigma = sigma - beta
weights = weights - multiply(add(subtract(beta, sigma), beta), constant(5.07243606e-02))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: alpha, lambda shape, alpha, beta, grad: tf.math.add(tf.math.multiply(tf.math.negative(grad), tf.math.pow(tf.math.sqrt(tf.math.square(tf.math.add(alpha, tf.math.square(tf.constant(9.94242714e-01, dtype=tf.float32))))), tf.math.square(grad))), beta), lambda shape, alpha, beta, sigma, grad: beta, lambda shape, alpha, beta, sigma, grad: tf.math.multiply(tf.math.add(tf.math.subtract(beta, sigma), beta), tf.constant(5.07243606e-02, dtype=tf.float32))
```
- `multi_no_arch_update_phen_id_44052`
  - criterion: `fmnist_score` = `0.8443999886512756`
  - adjusted_score: `2.3130666315555573`
  - task_score_total: `2.3130666315555573`
  - overall_score: `2.6582222233215966`; fitness: `-2.6582222233215966`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_2` / `5667` / `300`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.7706666588783264, cifar100=0.5923333168029785, fmnist=0.8443999886512756, tiny_imagenet=0.10566666722297668`
  - operation/source: `mutation` / `iteration_300.json`
  - smart_phenotype: `multiply(negative(add(multiply(subtract(grad, beta), add(constant(1.27951705e-01), grad)), grad)), sqrt(constant(1.90885420e-02)))`
  - advanced_readable_phenotype:

```text
beta = beta - negative(add(multiply(subtract(grad, beta), add(constant(1.27951705e-01), grad)), grad))
weights = weights - multiply(beta, sqrt(constant(1.90885420e-02)))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: grad, lambda shape, alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(grad, beta), tf.math.add(tf.constant(1.27951705e-01, dtype=tf.float32), grad)), grad)), lambda shape, alpha, beta, sigma, grad: tf.math.pow(tf.math.subtract(tf.math.sqrt(tf.constant(2.11963334e-01, dtype=tf.float32)), grad), grad), lambda shape, alpha, beta, sigma, grad: tf.math.multiply(beta, tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)))
```
- `multi_no_arch_update_phen_id_33247`
  - criterion: `fmnist_score` = `0.8441636562347412`
  - adjusted_score: `1.233496993780136`
  - task_score_total: `1.233496993780136`
  - overall_score: `1.495333343744278`; fitness: `-1.495333343744278`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_3` / `3180` / `130`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.3893333375453949, fmnist=0.8441636562347412`
  - operation/source: `mutation` / `iteration_130.json`
  - smart_phenotype: `multiply(add(subtract(add(multiply(grad, negative(pow(sqrt(square(add(alpha, square(constant(1.27951705e-01))))), square(alpha)))), beta), grad), add(multiply(grad, negative(pow(sqrt(square(add(alpha, square(constant(1.27951705e-01))))), square(alpha)))), beta)), constant(5.07243606e-02))`
  - advanced_readable_phenotype:

```text
alpha = alpha - alpha
beta = beta - add(multiply(grad, negative(pow(sqrt(square(add(alpha, square(constant(1.27951705e-01))))), square(alpha)))), beta)
sigma = sigma - grad
weights = weights - multiply(add(subtract(beta, sigma), beta), constant(5.07243606e-02))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: alpha, lambda shape, alpha, beta, grad: tf.math.add(tf.math.multiply(grad, tf.math.negative(tf.math.pow(tf.math.sqrt(tf.math.square(tf.math.add(alpha, tf.math.square(tf.constant(1.27951705e-01, dtype=tf.float32))))), tf.math.square(alpha)))), beta), lambda shape, alpha, beta, sigma, grad: grad, lambda shape, alpha, beta, sigma, grad: tf.math.multiply(tf.math.add(tf.math.subtract(beta, sigma), beta), tf.constant(5.07243606e-02, dtype=tf.float32))
```
- `multi_no_arch_update_phen_id_33120`
  - criterion: `fmnist_score` = `0.8438363671302795`
  - adjusted_score: `1.3808363676071167`
  - task_score_total: `1.3808363676071167`
  - overall_score: `1.2557103037834167`; fitness: `-1.2557103037834167`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_3` / `4697` / `167`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.5370000004768372, fmnist=0.8438363671302795`
  - operation/source: `mutation` / `iteration_167.json`
  - smart_phenotype: `multiply(add(subtract(add(multiply(grad, negative(pow(sqrt(square(add(add(grad, alpha), square(add(grad, alpha))))), square(add(grad, alpha))))), beta), grad), add(multiply(grad, negative(pow(sqrt(square(add(add(grad, alpha), square(add(grad, alpha))))), square(add(grad, alpha))))), beta)), constant(5.07243606e-02))`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(grad, alpha)
beta = beta - add(multiply(grad, negative(pow(sqrt(square(add(alpha, square(alpha)))), square(alpha)))), beta)
sigma = sigma - grad
weights = weights - multiply(add(subtract(beta, sigma), beta), constant(5.07243606e-02))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: tf.math.add(grad, alpha), lambda shape, alpha, beta, grad: tf.math.add(tf.math.multiply(grad, tf.math.negative(tf.math.pow(tf.math.sqrt(tf.math.square(tf.math.add(alpha, tf.math.square(alpha)))), tf.math.square(alpha)))), beta), lambda shape, alpha, beta, sigma, grad: grad, lambda shape, alpha, beta, sigma, grad: tf.math.multiply(tf.math.add(tf.math.subtract(beta, sigma), beta), tf.constant(5.07243606e-02, dtype=tf.float32))
```
- `multi_no_arch_update_phen_id_32961`
  - criterion: `fmnist_score` = `0.8435272574424744`
  - adjusted_score: `1.3315272629261017`
  - task_score_total: `1.3315272629261017`
  - overall_score: `1.2118509113788605`; fitness: `-1.2118509113788605`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_3` / `4037` / `152`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.4880000054836273, fmnist=0.8435272574424744`
  - operation/source: `mutation` / `iteration_152.json`
  - smart_phenotype: `multiply(add(subtract(add(multiply(grad, negative(pow(sqrt(add(add(alpha, square(alpha)), square(grad))), alpha))), beta), grad), add(multiply(grad, negative(pow(sqrt(add(add(alpha, square(alpha)), square(grad))), alpha))), beta)), constant(5.07243606e-02))`
  - advanced_readable_phenotype:

```text
alpha = alpha - alpha
beta = beta - add(multiply(grad, negative(pow(sqrt(add(add(alpha, square(alpha)), square(grad))), alpha))), beta)
sigma = sigma - grad
weights = weights - multiply(add(subtract(beta, sigma), beta), constant(5.07243606e-02))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: alpha, lambda shape, alpha, beta, grad: tf.math.add(tf.math.multiply(grad, tf.math.negative(tf.math.pow(tf.math.sqrt(tf.math.add(tf.math.add(alpha, tf.math.square(alpha)), tf.math.square(grad))), alpha))), beta), lambda shape, alpha, beta, sigma, grad: grad, lambda shape, alpha, beta, sigma, grad: tf.math.multiply(tf.math.add(tf.math.subtract(beta, sigma), beta), tf.constant(5.07243606e-02, dtype=tf.float32))
```
- `multi_no_arch_update_phen_id_34294`
  - criterion: `fmnist_score` = `0.8434000015258789`
  - adjusted_score: `1.312066674232483`
  - task_score_total: `1.312066674232483`
  - overall_score: `1.2722654536366462`; fitness: `-1.2722654536366462`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_3` / `3710` / `144`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.468666672706604, fmnist=0.8434000015258789`
  - operation/source: `mutation` / `iteration_144.json`
  - smart_phenotype: `multiply(add(subtract(add(multiply(grad, negative(pow(sqrt(square(add(grad, square(grad)))), square(alpha)))), beta), grad), add(multiply(grad, negative(pow(sqrt(square(add(grad, square(grad)))), square(alpha)))), beta)), constant(5.07243606e-02))`
  - advanced_readable_phenotype:

```text
alpha = alpha - alpha
beta = beta - add(multiply(grad, negative(pow(sqrt(square(add(grad, square(grad)))), square(alpha)))), beta)
sigma = sigma - grad
weights = weights - multiply(add(subtract(beta, sigma), beta), constant(5.07243606e-02))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: alpha, lambda shape, alpha, beta, grad: tf.math.add(tf.math.multiply(grad, tf.math.negative(tf.math.pow(tf.math.sqrt(tf.math.square(tf.math.add(grad, tf.math.square(grad)))), tf.math.square(alpha)))), beta), lambda shape, alpha, beta, sigma, grad: grad, lambda shape, alpha, beta, sigma, grad: tf.math.multiply(tf.math.add(tf.math.subtract(beta, sigma), beta), tf.constant(5.07243606e-02, dtype=tf.float32))
```
- `multi_no_arch_update_phen_id_33446`
  - criterion: `fmnist_score` = `0.8429090976715088`
  - adjusted_score: `1.2825757563114166`
  - task_score_total: `1.2825757563114166`
  - overall_score: `1.3952222168445587`; fitness: `-1.3952222168445587`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_3` / `5386` / `180`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.43966665863990784, fmnist=0.8429090976715088`
  - operation/source: `mutation` / `iteration_180.json`
  - smart_phenotype: `multiply(add(subtract(add(multiply(grad, negative(pow(sqrt(square(add(alpha, square(grad)))), square(alpha)))), beta), grad), add(multiply(grad, negative(pow(sqrt(square(add(alpha, square(grad)))), square(alpha)))), beta)), constant(5.07243606e-02))`
  - advanced_readable_phenotype:

```text
alpha = alpha - alpha
beta = beta - add(multiply(grad, negative(pow(sqrt(square(add(alpha, square(grad)))), square(alpha)))), beta)
sigma = sigma - grad
weights = weights - multiply(add(subtract(beta, sigma), beta), constant(5.07243606e-02))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: alpha, lambda shape, alpha, beta, grad: tf.math.add(tf.math.multiply(grad, tf.math.negative(tf.math.pow(tf.math.sqrt(tf.math.square(tf.math.add(alpha, tf.math.square(grad)))), tf.math.square(alpha)))), beta), lambda shape, alpha, beta, sigma, grad: grad, lambda shape, alpha, beta, sigma, grad: tf.math.multiply(tf.math.add(tf.math.subtract(beta, sigma), beta), tf.constant(5.07243606e-02, dtype=tf.float32))
```
- `multi_no_arch_update_phen_id_33009`
  - criterion: `fmnist_score` = `0.8425454497337341`
  - adjusted_score: `1.3442121148109436`
  - task_score_total: `1.3442121148109436`
  - overall_score: `1.2640080849329631`; fitness: `-1.2640080849329631`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_3` / `4351` / `160`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.5016666650772095, fmnist=0.8425454497337341`
  - operation/source: `mutation` / `iteration_160.json`
  - smart_phenotype: `multiply(add(subtract(add(multiply(grad, negative(pow(sqrt(divide_no_nan(add(grad, square(square(square(grad)))), square(grad))), divide_no_nan(square(add(alpha, alpha)), subtract(constant(2.11963334e-01), grad))))), beta), grad), add(multiply(grad, negative(pow(sqrt(divide_no_nan(add(grad, square(square(square(grad)))), square(grad))), divide_no_nan(square(add(alpha, alpha)), subtract(constant(2.11963334e-01), grad))))), beta)), constant(4.18339400e-02))`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(alpha, alpha)
beta = beta - add(multiply(grad, negative(pow(sqrt(divide_no_nan(add(grad, square(square(square(grad)))), square(grad))), divide_no_nan(square(alpha), subtract(constant(2.11963334e-01), grad))))), beta)
sigma = sigma - grad
weights = weights - multiply(add(subtract(beta, sigma), beta), constant(4.18339400e-02))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: tf.math.add(alpha, alpha), lambda shape, alpha, beta, grad: tf.math.add(tf.math.multiply(grad, tf.math.negative(tf.math.pow(tf.math.sqrt(tf.math.divide_no_nan(tf.math.add(grad, tf.math.square(tf.math.square(tf.math.square(grad)))), tf.math.square(grad))), tf.math.divide_no_nan(tf.math.square(alpha), tf.math.subtract(tf.constant(2.11963334e-01, dtype=tf.float32), grad))))), beta), lambda shape, alpha, beta, sigma, grad: grad, lambda shape, alpha, beta, sigma, grad: tf.math.multiply(tf.math.add(tf.math.subtract(beta, sigma), beta), tf.constant(4.18339400e-02, dtype=tf.float32))
```
- `multi_no_arch_update_phen_id_33124`
  - criterion: `fmnist_score` = `0.8425090909004211`
  - adjusted_score: `1.297842413187027`
  - task_score_total: `1.297842413187027`
  - overall_score: `1.329496967792511`; fitness: `-1.329496967792511`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_3` / `4387` / `161`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.45533332228660583, fmnist=0.8425090909004211`
  - operation/source: `mutation` / `iteration_161.json`
  - smart_phenotype: `multiply(add(subtract(add(multiply(grad, negative(pow(sqrt(square(add(add(grad, alpha), square(beta)))), square(grad)))), beta), grad), add(multiply(grad, negative(pow(sqrt(square(add(add(grad, alpha), square(beta)))), square(grad)))), beta)), constant(4.18339400e-02))`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(grad, alpha)
beta = beta - add(multiply(grad, negative(pow(sqrt(square(add(alpha, square(beta)))), square(grad)))), beta)
sigma = sigma - grad
weights = weights - multiply(add(subtract(beta, sigma), beta), constant(4.18339400e-02))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: tf.math.add(grad, alpha), lambda shape, alpha, beta, grad: tf.math.add(tf.math.multiply(grad, tf.math.negative(tf.math.pow(tf.math.sqrt(tf.math.square(tf.math.add(alpha, tf.math.square(beta)))), tf.math.square(grad)))), beta), lambda shape, alpha, beta, sigma, grad: grad, lambda shape, alpha, beta, sigma, grad: tf.math.multiply(tf.math.add(tf.math.subtract(beta, sigma), beta), tf.constant(4.18339400e-02, dtype=tf.float32))
```

## Cross-Setup Task: tiny_imagenet

- `no_multi_no_arch_update_phen_id_58316`
  - criterion: `tiny_imagenet_score` = `0.30106666684150696`
  - adjusted_score: `2.441196666841507`
  - task_score_total: `None`
  - overall_score: `0.30106666684150696`; fitness: `-0.30106666684150696`
  - setup/run/genetic/generation: `no_multi_no_arch_update` / `run_3` / `514` / `20`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.30106666684150696`
  - operation/source: `mutation` / `iteration_20.json`
  - smart_phenotype: `add(subtract(add(multiply(constant(1.07052146e-01), alpha), alpha), multiply(add(subtract(negative(sqrt(add(multiply(constant(1.07052146e-01), alpha), alpha))), sigma), grad), constant(4.74768924e-01))), sigma)`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(multiply(constant(1.07052146e-01), alpha), alpha)
sigma = sigma - add(subtract(alpha, multiply(add(subtract(negative(sqrt(alpha)), sigma), grad), constant(4.74768924e-01))), sigma)
weights = weights - sigma

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: tf.math.add(tf.math.multiply(tf.constant(1.07052146e-01, dtype=tf.float32), alpha), alpha), lambda shape, alpha, beta, grad: tf.math.add(tf.math.negative(tf.math.multiply(tf.math.divide_no_nan(grad, tf.math.subtract(tf.math.add(tf.constant(9.97895596e-01, dtype=tf.float32), tf.constant(9.96851186e-01, dtype=tf.float32)), tf.math.add(grad, tf.math.subtract(grad, tf.math.square(tf.math.add(alpha, tf.constant(9.98594080e-01, dtype=tf.float32))))))), tf.constant(9.94242714e-01, dtype=tf.float32))), beta), lambda shape, alpha, beta, sigma, grad: tf.math.add(tf.math.subtract(alpha, tf.math.multiply(tf.math.add(tf.math.subtract(tf.math.negative(tf.math.sqrt(alpha)), sigma), grad), tf.constant(4.74768924e-01, dtype=tf.float32))), sigma), lambda shape, alpha, beta, sigma, grad: sigma
```
- `no_multi_no_arch_update_phen_id_57728`
  - criterion: `tiny_imagenet_score` = `0.29546666145324707`
  - adjusted_score: `2.435596661453247`
  - task_score_total: `None`
  - overall_score: `0.29546666145324707`; fitness: `-0.29546666145324707`
  - setup/run/genetic/generation: `no_multi_no_arch_update` / `run_3` / `428` / `10`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.29546666145324707`
  - operation/source: `mutation` / `iteration_10.json`
  - smart_phenotype: `add(add(multiply(grad, constant(1.07052146e-01)), alpha), sigma)`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(multiply(grad, constant(1.07052146e-01)), alpha)
sigma = sigma - add(alpha, sigma)
weights = weights - sigma

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: tf.math.add(tf.math.multiply(grad, tf.constant(1.07052146e-01, dtype=tf.float32)), alpha), lambda shape, alpha, beta, grad: tf.math.add(tf.math.negative(tf.math.multiply(tf.math.divide_no_nan(tf.constant(9.97895596e-01, dtype=tf.float32), tf.math.subtract(tf.math.add(tf.constant(9.96851186e-01, dtype=tf.float32), alpha), tf.math.add(grad, tf.math.subtract(grad, tf.math.square(tf.math.add(tf.constant(9.98594080e-01, dtype=tf.float32), tf.constant(9.94242714e-01, dtype=tf.float32))))))), tf.constant(9.99944439e-01, dtype=tf.float32))), beta), lambda shape, alpha, beta, sigma, grad: tf.math.add(alpha, sigma), lambda shape, alpha, beta, sigma, grad: sigma
```
- `no_multi_arch_update_phen_id_52872`
  - criterion: `tiny_imagenet_score` = `0.2944000005722046`
  - adjusted_score: `2.4345300005722046`
  - task_score_total: `None`
  - overall_score: `0.2944000005722046`; fitness: `-0.2944000005722046`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_4` / `809` / `18`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `tiny_imagenet=0.2944000005722046`
  - operation/source: `mutation` / `iteration_18.json`
  - smart_phenotype: `divide_no_nan(add(negative(grad), beta), sqrt(layer_count))`
  - advanced_readable_phenotype:

```text
beta = beta - add(negative(grad), beta)
weights = weights - divide_no_nan(beta, sqrt(layer_count))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: alpha, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.negative(grad), beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(padding, sigma), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.divide_no_nan(beta, tf.math.sqrt(layer_count))
```
- `no_multi_arch_update_phen_id_52912`
  - criterion: `tiny_imagenet_score` = `0.2933999955654144`
  - adjusted_score: `2.4335299955654146`
  - task_score_total: `None`
  - overall_score: `0.2933999955654144`; fitness: `-0.2933999955654144`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_4` / `1104` / `77`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `tiny_imagenet=0.2933999955654144`
  - operation/source: `mutation` / `iteration_77.json`
  - smart_phenotype: `divide_no_nan(add(negative(subtract(grad, multiply(subtract(divide_no_nan(layer_num, add(negative(layer_count), padding)), add(constant(9.76737464e-01), dilation_rate)), grad))), beta), sqrt(layer_count))`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(negative(layer_count), padding)
beta = beta - add(negative(subtract(grad, multiply(subtract(divide_no_nan(layer_num, alpha), add(constant(9.76737464e-01), dilation_rate)), grad))), beta)
weights = weights - divide_no_nan(beta, sqrt(layer_count))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(tf.math.negative(layer_count), padding), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.negative(tf.math.subtract(grad, tf.math.multiply(tf.math.subtract(tf.math.divide_no_nan(layer_num, alpha), tf.math.add(tf.constant(9.76737464e-01, dtype=tf.float32), dilation_rate)), grad))), beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: layer_num, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.divide_no_nan(beta, tf.math.sqrt(layer_count))
```
- `no_multi_arch_update_phen_id_52868`
  - criterion: `tiny_imagenet_score` = `0.2931333303451538`
  - adjusted_score: `2.433263330345154`
  - task_score_total: `None`
  - overall_score: `0.2931333303451538`; fitness: `-0.2931333303451538`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_4` / `1006` / `49`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `tiny_imagenet=0.2931333303451538`
  - operation/source: `mutation` / `iteration_49.json`
  - smart_phenotype: `divide_no_nan(add(negative(grad), beta), sqrt(add(layer_count, add(layer_num, sigma))))`
  - advanced_readable_phenotype:

```text
beta = beta - add(negative(grad), beta)
sigma = sigma - add(layer_num, sigma)
weights = weights - divide_no_nan(beta, sqrt(add(layer_count, sigma)))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: grad, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.negative(grad), beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(layer_num, sigma), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.divide_no_nan(beta, tf.math.sqrt(tf.math.add(layer_count, sigma)))
```
- `multi_no_arch_update_phen_id_42625`
- `multi_no_arch_update_phen_id_45499`
- `no_multi_arch_update_phen_id_52874`
  - criterion: `tiny_imagenet_score` = `0.2859333395957947`
  - adjusted_score: `2.4260633395957947`
  - task_score_total: `None`
  - overall_score: `0.2859333395957947`; fitness: `-0.2859333395957947`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_4` / `911` / `32`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `tiny_imagenet=0.2859333395957947`
  - operation/source: `mutation` / `iteration_32.json`
  - smart_phenotype: `divide_no_nan(add(negative(grad), beta), sqrt(subtract(layer_count, subtract(subtract(layer_num, pow(add(padding, sigma), layer_num)), kernel_size))))`
  - advanced_readable_phenotype:

```text
beta = beta - add(negative(grad), beta)
sigma = sigma - add(padding, sigma)
weights = weights - divide_no_nan(beta, sqrt(subtract(layer_count, subtract(subtract(layer_num, pow(sigma, layer_num)), kernel_size))))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(tf.constant(5.75183132e-01, dtype=tf.float32), alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.negative(grad), beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(padding, sigma), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.divide_no_nan(beta, tf.math.sqrt(tf.math.subtract(layer_count, tf.math.subtract(tf.math.subtract(layer_num, tf.math.pow(sigma, layer_num)), kernel_size))))
```
- `multi_arch_update_phen_id_22729`
- `no_multi_arch_update_phen_id_56942`
  - criterion: `tiny_imagenet_score` = `0.28173333406448364`
  - adjusted_score: `2.4218633340644837`
  - task_score_total: `None`
  - overall_score: `0.28173333406448364`; fitness: `-0.28173333406448364`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_3` / `677` / `11`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `tiny_imagenet=0.28173333406448364`
  - operation/source: `mutation` / `iteration_11.json`
  - smart_phenotype: `subtract(add(add(multiply(negative(square(add(pool_size, beta))), sigma), sigma), add(pool_size, beta)), add(grad, alpha))`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(grad, alpha)
beta = beta - add(pool_size, beta)
sigma = sigma - add(multiply(negative(square(beta)), sigma), sigma)
weights = weights - subtract(add(sigma, beta), alpha)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(grad, alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(pool_size, beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(tf.math.multiply(tf.math.negative(tf.math.square(beta)), sigma), sigma), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.subtract(tf.math.add(sigma, beta), alpha)
```

## multi_arch_update

### Adjusted Overall

- `multi_arch_update_phen_id_22661`
- `multi_arch_update_phen_id_22676`
- `multi_arch_update_phen_id_22729`
- `multi_arch_update_phen_id_22702`
- `multi_arch_update_phen_id_22713`
- `multi_arch_update_phen_id_22716`
- `multi_arch_update_phen_id_22865`
  - criterion: `adjusted_score` = `1.4232909083366394`
  - adjusted_score: `1.4232909083366394`
  - task_score_total: `1.4232909083366394`
  - overall_score: `0.9683236360549927`; fitness: `-0.9683236360549927`
  - setup/run/genetic/generation: `multi_arch_update` / `run_4` / `1767` / `110`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `cifar10=0.5899999737739563, fmnist=0.8332909345626831`
  - operation/source: `mutation` / `iteration_110.json`
  - smart_phenotype: `negative(divide_no_nan(add(multiply(sqrt(constant(2.28478855e-04)), add(beta, grad)), grad), layer_count))`
  - advanced_readable_phenotype:

```text
beta = beta - add(multiply(sqrt(constant(2.28478855e-04)), add(beta, grad)), grad)
weights = weights - negative(divide_no_nan(beta, layer_count))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(tf.math.square(layer_count), grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.multiply(tf.math.sqrt(tf.constant(2.28478855e-04, dtype=tf.float32)), tf.math.add(beta, grad)), grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(grad, layer_count), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(tf.math.divide_no_nan(beta, layer_count))
```
- `multi_arch_update_phen_id_19238`
  - criterion: `adjusted_score` = `1.419903039932251`
  - adjusted_score: `1.419903039932251`
  - task_score_total: `1.419903039932251`
  - overall_score: `0.9602606058120727`; fitness: `-0.9602606058120727`
  - setup/run/genetic/generation: `multi_arch_update` / `run_2` / `2350` / `99`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `cifar10=0.5876666903495789, fmnist=0.8322363495826721`
  - operation/source: `mutation` / `iteration_99.json`
  - smart_phenotype: `divide_no_nan(square(negative(divide_no_nan(grad, add(grad, add(layer_count, subtract(divide_no_nan(square(add(divide_no_nan(pool_size, square(constant(9.99487531e-01))), strides)), add(pool_size, constant(6.27124987e-04))), grad)))))), negative(divide_no_nan(grad, add(grad, add(layer_count, subtract(divide_no_nan(square(add(divide_no_nan(pool_size, square(constant(9.99487531e-01))), strides)), add(pool_size, constant(6.27124987e-04))), grad))))))`
  - advanced_readable_phenotype:

```text
beta = beta - negative(divide_no_nan(grad, add(grad, add(layer_count, subtract(divide_no_nan(square(add(divide_no_nan(pool_size, square(constant(9.99487531e-01))), strides)), add(pool_size, constant(6.27124987e-04))), grad)))))
weights = weights - divide_no_nan(square(beta), beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: dilation_rate, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.negative(tf.math.divide_no_nan(grad, tf.math.add(grad, tf.math.add(layer_count, tf.math.subtract(tf.math.divide_no_nan(tf.math.square(tf.math.add(tf.math.divide_no_nan(pool_size, tf.math.square(tf.constant(9.99487531e-01, dtype=tf.float32))), strides)), tf.math.add(pool_size, tf.constant(6.27124987e-04, dtype=tf.float32))), grad))))), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: layer_num, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.divide_no_nan(tf.math.square(beta), beta)
```
- `multi_arch_update_phen_id_17945`
  - criterion: `adjusted_score` = `1.3836302757263184`
  - adjusted_score: `1.3836302757263184`
  - task_score_total: `1.3836302757263184`
  - overall_score: `1.1015308976173401`; fitness: `-1.1015308976173401`
  - setup/run/genetic/generation: `multi_arch_update` / `run_2` / `7719` / `447`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `cifar10=0.5496666431427002, fmnist=0.8339636325836182`
  - operation/source: `mutation` / `iteration_447.json`
  - smart_phenotype: `divide_no_nan(square(negative(divide_no_nan(add(grad, grad), add(layer_count, subtract(divide_no_nan(square(add(divide_no_nan(pool_size, square(constant(2.79614739e-04))), filters)), add(pool_size, strides)), grad))))), negative(divide_no_nan(add(grad, grad), add(layer_count, subtract(divide_no_nan(square(add(divide_no_nan(pool_size, square(constant(2.79614739e-04))), filters)), add(pool_size, strides)), grad)))))`
  - advanced_readable_phenotype:

```text
beta = beta - negative(divide_no_nan(add(grad, grad), add(layer_count, subtract(divide_no_nan(square(add(divide_no_nan(pool_size, square(constant(2.79614739e-04))), filters)), add(pool_size, strides)), grad))))
weights = weights - divide_no_nan(square(beta), beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: grad, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.negative(tf.math.divide_no_nan(tf.math.add(grad, grad), tf.math.add(layer_count, tf.math.subtract(tf.math.divide_no_nan(tf.math.square(tf.math.add(tf.math.divide_no_nan(pool_size, tf.math.square(tf.constant(2.79614739e-04, dtype=tf.float32))), filters)), tf.math.add(pool_size, strides)), grad)))), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.multiply(tf.math.square(layer_num), tf.math.pow(tf.math.add(layer_count, layer_count), grad)), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.divide_no_nan(tf.math.square(beta), beta)
```
- `multi_arch_update_phen_id_17876`
  - criterion: `adjusted_score` = `1.3815696835517883`
  - adjusted_score: `1.3815696835517883`
  - task_score_total: `1.3815696835517883`
  - overall_score: `0.9531321167945862`; fitness: `-0.9531321167945862`
  - setup/run/genetic/generation: `multi_arch_update` / `run_2` / `5911` / `351`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `cifar10=0.5503333210945129, fmnist=0.8312363624572754`
  - operation/source: `mutation` / `iteration_351.json`
  - smart_phenotype: `divide_no_nan(square(negative(divide_no_nan(add(grad, grad), add(layer_count, subtract(divide_no_nan(square(add(divide_no_nan(filters, strides), square(pool_size))), constant(9.99372875e-01)), beta))))), negative(divide_no_nan(add(grad, grad), add(layer_count, subtract(divide_no_nan(square(add(divide_no_nan(filters, strides), square(pool_size))), constant(9.99372875e-01)), beta)))))`
  - advanced_readable_phenotype:

```text
beta = beta - negative(divide_no_nan(add(grad, grad), add(layer_count, subtract(divide_no_nan(square(add(divide_no_nan(filters, strides), square(pool_size))), constant(9.99372875e-01)), beta))))
weights = weights - divide_no_nan(square(beta), beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: pool_size, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.negative(tf.math.divide_no_nan(tf.math.add(grad, grad), tf.math.add(layer_count, tf.math.subtract(tf.math.divide_no_nan(tf.math.square(tf.math.add(tf.math.divide_no_nan(filters, strides), tf.math.square(pool_size))), tf.constant(9.99372875e-01, dtype=tf.float32)), beta)))), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.square(tf.math.square(layer_num)), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.divide_no_nan(tf.math.square(beta), beta)
```

### Task-Score Overall

- `multi_arch_update_phen_id_22661`
- `multi_arch_update_phen_id_22676`
- `multi_arch_update_phen_id_22729`
- `multi_arch_update_phen_id_22702`
- `multi_arch_update_phen_id_22713`
- `multi_arch_update_phen_id_22716`
- `multi_arch_update_phen_id_22865`
- `multi_arch_update_phen_id_19238`
- `multi_arch_update_phen_id_17945`
- `multi_arch_update_phen_id_17876`

### With Architecture

- `multi_arch_update_phen_id_22661`
- `multi_arch_update_phen_id_22676`
- `multi_arch_update_phen_id_22729`
- `multi_arch_update_phen_id_22702`
- `multi_arch_update_phen_id_22713`
- `multi_arch_update_phen_id_22716`
- `multi_arch_update_phen_id_22865`
- `multi_arch_update_phen_id_19238`
- `multi_arch_update_phen_id_17945`
- `multi_arch_update_phen_id_17876`

### Without Architecture

- `multi_arch_update_phen_id_25520`
  - criterion: `adjusted_score` = `0.8033999800682068`
  - adjusted_score: `0.8033999800682068`
  - task_score_total: `0.8033999800682068`
  - overall_score: `0.7832945466041565`; fitness: `-0.7832945466041565`
  - setup/run/genetic/generation: `multi_arch_update` / `run_4` / `1022` / `35`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `fmnist=0.8033999800682068`
  - operation/source: `mutation` / `iteration_35.json`
  - smart_phenotype: `negative(multiply(constant(1.90885420e-02), grad))`
  - advanced_readable_phenotype:

```text
beta = beta - multiply(constant(1.90885420e-02), grad)
weights = weights - negative(beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.divide_no_nan(filters, tf.math.add(tf.math.pow(tf.constant(9.99944439e-01, dtype=tf.float32), tf.math.sqrt(tf.math.add(grad, layer_count))), tf.constant(9.98279874e-01, dtype=tf.float32))), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.multiply(tf.constant(1.90885420e-02, dtype=tf.float32), grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.constant(3.76354517e-01, dtype=tf.float32), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(beta)
```
- `multi_arch_update_phen_id_25486`
  - criterion: `adjusted_score` = `0.7814727425575256`
  - adjusted_score: `0.7814727425575256`
  - task_score_total: `0.7814727425575256`
  - overall_score: `0.7830763697624207`; fitness: `-0.7830763697624207`
  - setup/run/genetic/generation: `multi_arch_update` / `run_4` / `850` / `24`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `fmnist=0.7814727425575256`
  - operation/source: `mutation` / `iteration_24.json`
  - smart_phenotype: `negative(multiply(add(constant(1.90885420e-02), beta), grad))`
  - advanced_readable_phenotype:

```text
beta = beta - multiply(add(constant(1.90885420e-02), beta), grad)
weights = weights - negative(beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.divide_no_nan(layer_num, grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.multiply(tf.math.add(tf.constant(1.90885420e-02, dtype=tf.float32), beta), grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: sigma, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(beta)
```
- `multi_arch_update_phen_id_25557`
  - criterion: `adjusted_score` = `0.7738727331161499`
  - adjusted_score: `0.7738727331161499`
  - task_score_total: `0.7738727331161499`
  - overall_score: `0.7698290824890137`; fitness: `-0.7698290824890137`
  - setup/run/genetic/generation: `multi_arch_update` / `run_4` / `1163` / `44`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `fmnist=0.7738727331161499`
  - operation/source: `mutation` / `iteration_44.json`
  - smart_phenotype: `negative(multiply(multiply(constant(1.90885420e-02), constant(9.99720385e-01)), grad))`
  - advanced_readable_phenotype:

```text
beta = beta - multiply(multiply(constant(1.90885420e-02), constant(9.99720385e-01)), grad)
weights = weights - negative(beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.divide_no_nan(grad, grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.multiply(tf.math.multiply(tf.constant(1.90885420e-02, dtype=tf.float32), tf.constant(9.99720385e-01, dtype=tf.float32)), grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: layer_num, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(beta)
```
- `multi_arch_update_phen_id_25489`
  - criterion: `adjusted_score` = `0.7666181921958923`
  - adjusted_score: `0.7666181921958923`
  - task_score_total: `0.7666181921958923`
  - overall_score: `0.7745454549789429`; fitness: `-0.7745454549789429`
  - setup/run/genetic/generation: `multi_arch_update` / `run_4` / `1021` / `35`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `fmnist=0.7666181921958923`
  - operation/source: `mutation` / `iteration_35.json`
  - smart_phenotype: `negative(multiply(add(constant(1.90885420e-02), negative(beta)), grad))`
  - advanced_readable_phenotype:

```text
beta = beta - multiply(add(constant(1.90885420e-02), negative(beta)), grad)
weights = weights - negative(beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(tf.math.divide_no_nan(layer_num, grad), alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.multiply(tf.math.add(tf.constant(1.90885420e-02, dtype=tf.float32), tf.math.negative(beta)), grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: sigma, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(beta)
```
- `multi_arch_update_phen_id_3032`
  - criterion: `adjusted_score` = `0.749963641166687`
  - adjusted_score: `0.749963641166687`
  - task_score_total: `0.749963641166687`
  - overall_score: `0.7204800009727478`; fitness: `-0.7204800009727478`
  - setup/run/genetic/generation: `multi_arch_update` / `run_2` / `769` / `22`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `fmnist=0.749963641166687`
  - operation/source: `mutation` / `iteration_22.json`
  - smart_phenotype: `add(negative(multiply(add(grad, constant(6.13831074e-02)), grad)), beta)`
  - advanced_readable_phenotype:

```text
beta = beta - add(negative(multiply(add(grad, constant(6.13831074e-02)), grad)), beta)
weights = weights - beta

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: layer_num, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.negative(tf.math.multiply(tf.math.add(grad, tf.constant(6.13831074e-02, dtype=tf.float32)), grad)), beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.constant(9.99875353e-01, dtype=tf.float32), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: beta
```
- `multi_arch_update_phen_id_25638`
  - criterion: `adjusted_score` = `0.7466909289360046`
  - adjusted_score: `0.7466909289360046`
  - task_score_total: `0.7466909289360046`
  - overall_score: `0.7374909122784933`; fitness: `-0.7374909122784933`
  - setup/run/genetic/generation: `multi_arch_update` / `run_4` / `1183` / `46`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `fmnist=0.7466909289360046`
  - operation/source: `mutation` / `iteration_46.json`
  - smart_phenotype: `negative(multiply(multiply(grad, constant(1.90885420e-02)), constant(4.74768924e-01)))`
  - advanced_readable_phenotype:

```text
beta = beta - multiply(multiply(grad, constant(1.90885420e-02)), constant(4.74768924e-01))
weights = weights - negative(beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(tf.math.add(filters, grad), alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.multiply(tf.math.multiply(grad, tf.constant(1.90885420e-02, dtype=tf.float32)), tf.constant(4.74768924e-01, dtype=tf.float32)), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: filters, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(beta)
```
- `multi_arch_update_phen_id_20626`
  - criterion: `adjusted_score` = `0.7340182065963745`
  - adjusted_score: `0.7340182065963745`
  - task_score_total: `0.7340182065963745`
  - overall_score: `0.7323545515537262`; fitness: `-0.7323545515537262`
  - setup/run/genetic/generation: `multi_arch_update` / `run_4` / `840` / `23`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `fmnist=0.7340182065963745`
  - operation/source: `mutation` / `iteration_23.json`
  - smart_phenotype: `negative(add(multiply(add(constant(1.90885420e-02), beta), grad), beta))`
  - advanced_readable_phenotype:

```text
beta = beta - add(multiply(add(constant(1.90885420e-02), beta), grad), beta)
weights = weights - negative(beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.divide_no_nan(layer_num, grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.multiply(tf.math.add(tf.constant(1.90885420e-02, dtype=tf.float32), beta), grad), beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: sigma, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(beta)
```
- `multi_arch_update_phen_id_20670`
  - criterion: `adjusted_score` = `0.7337454557418823`
  - adjusted_score: `0.7337454557418823`
  - task_score_total: `0.7337454557418823`
  - overall_score: `0.7305772751569748`; fitness: `-0.7305772751569748`
  - setup/run/genetic/generation: `multi_arch_update` / `run_4` / `841` / `23`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `fmnist=0.7337454557418823`
  - operation/source: `mutation` / `iteration_23.json`
  - smart_phenotype: `negative(add(multiply(multiply(constant(1.90885420e-02), constant(9.99813307e-01)), grad), beta))`
  - advanced_readable_phenotype:

```text
beta = beta - add(multiply(multiply(constant(1.90885420e-02), constant(9.99813307e-01)), grad), beta)
weights = weights - negative(beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.divide_no_nan(filters, layer_num), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.multiply(tf.math.multiply(tf.constant(1.90885420e-02, dtype=tf.float32), tf.constant(9.99813307e-01, dtype=tf.float32)), grad), beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: sigma, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(beta)
```
- `multi_arch_update_phen_id_20666`
  - criterion: `adjusted_score` = `0.7283272743225098`
  - adjusted_score: `0.7283272743225098`
  - task_score_total: `0.7283272743225098`
  - overall_score: `0.7159454425175985`; fitness: `-0.7159454425175985`
  - setup/run/genetic/generation: `multi_arch_update` / `run_4` / `895` / `27`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `fmnist=0.7283272743225098`
  - operation/source: `mutation` / `iteration_27.json`
  - smart_phenotype: `negative(add(multiply(multiply(constant(1.90885420e-02), constant(3.76354517e-01)), grad), beta))`
  - advanced_readable_phenotype:

```text
beta = beta - add(multiply(multiply(constant(1.90885420e-02), constant(3.76354517e-01)), grad), beta)
weights = weights - negative(beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.divide_no_nan(grad, filters), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.multiply(tf.math.multiply(tf.constant(1.90885420e-02, dtype=tf.float32), tf.constant(3.76354517e-01, dtype=tf.float32)), grad), beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: grad, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(beta)
```
- `multi_arch_update_phen_id_25556`
  - criterion: `adjusted_score` = `0.6722182035446167`
  - adjusted_score: `0.6722182035446167`
  - task_score_total: `0.6722182035446167`
  - overall_score: `0.6991515159606934`; fitness: `-0.6991515159606934`
  - setup/run/genetic/generation: `multi_arch_update` / `run_4` / `978` / `31`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `fmnist=0.6722182035446167`
  - operation/source: `mutation` / `iteration_31.json`
  - smart_phenotype: `negative(multiply(multiply(constant(1.90885420e-02), constant(6.13831074e-02)), grad))`
  - advanced_readable_phenotype:

```text
beta = beta - multiply(multiply(constant(1.90885420e-02), constant(6.13831074e-02)), grad)
weights = weights - negative(beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.divide_no_nan(filters, grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.multiply(tf.math.multiply(tf.constant(1.90885420e-02, dtype=tf.float32), tf.constant(6.13831074e-02, dtype=tf.float32)), grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: layer_count, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(beta)
```

### Task: cifar10

- `multi_arch_update_phen_id_22676`
- `multi_arch_update_phen_id_22713`
- `multi_arch_update_phen_id_22702`
- `multi_arch_update_phen_id_22661`
- `multi_arch_update_phen_id_22729`
- `multi_arch_update_phen_id_22716`
- `multi_arch_update_phen_id_22865`
- `multi_arch_update_phen_id_19238`
- `multi_arch_update_phen_id_17876`
- `multi_arch_update_phen_id_17945`

### Task: cifar100

- `multi_arch_update_phen_id_22661`
- `multi_arch_update_phen_id_22676`
- `multi_arch_update_phen_id_22702`
- `multi_arch_update_phen_id_22716`
- `multi_arch_update_phen_id_22729`
- `multi_arch_update_phen_id_22713`

### Task: fmnist

- `multi_arch_update_phen_id_17958`
  - criterion: `fmnist_score` = `0.840254545211792`
  - adjusted_score: `1.3515878915786743`
  - task_score_total: `1.3515878915786743`
  - overall_score: `1.0873139381408692`; fitness: `-1.0873139381408692`
  - setup/run/genetic/generation: `multi_arch_update` / `run_2` / `6578` / `387`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `cifar10=0.5113333463668823, fmnist=0.840254545211792`
  - operation/source: `mutation` / `iteration_387.json`
  - smart_phenotype: `divide_no_nan(square(negative(divide_no_nan(add(grad, grad), add(layer_count, subtract(divide_no_nan(square(add(divide_no_nan(pool_size, square(constant(5.75183132e-01))), kernel_size)), add(padding, constant(5.75728612e-03))), grad))))), negative(divide_no_nan(add(grad, grad), add(layer_count, subtract(divide_no_nan(square(add(divide_no_nan(pool_size, square(constant(5.75183132e-01))), kernel_size)), add(padding, constant(5.75728612e-03))), grad)))))`
  - advanced_readable_phenotype:

```text
beta = beta - negative(divide_no_nan(add(grad, grad), add(layer_count, subtract(divide_no_nan(square(add(divide_no_nan(pool_size, square(constant(5.75183132e-01))), kernel_size)), add(padding, constant(5.75728612e-03))), grad))))
weights = weights - divide_no_nan(square(beta), beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: grad, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.negative(tf.math.divide_no_nan(tf.math.add(grad, grad), tf.math.add(layer_count, tf.math.subtract(tf.math.divide_no_nan(tf.math.square(tf.math.add(tf.math.divide_no_nan(pool_size, tf.math.square(tf.constant(5.75183132e-01, dtype=tf.float32))), kernel_size)), tf.math.add(padding, tf.constant(5.75728612e-03, dtype=tf.float32))), grad)))), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.multiply(tf.math.square(layer_num), tf.math.pow(tf.math.add(layer_count, dilation_rate), grad)), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.divide_no_nan(tf.math.square(beta), beta)
```
- `multi_arch_update_phen_id_22533`
  - criterion: `fmnist_score` = `0.8368181586265564`
  - adjusted_score: `0.9838181585073471`
  - task_score_total: `0.9838181585073471`
  - overall_score: `0.8690945595502854`; fitness: `-0.8690945595502854`
  - setup/run/genetic/generation: `multi_arch_update` / `run_4` / `1151` / `43`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `cifar10=0.1469999998807907, fmnist=0.8368181586265564`
  - operation/source: `mutation` / `iteration_43.json`
  - smart_phenotype: `negative(divide_no_nan(add(multiply(constant(1.90885420e-02), beta), grad), layer_count))`
  - advanced_readable_phenotype:

```text
beta = beta - add(multiply(constant(1.90885420e-02), beta), grad)
weights = weights - negative(divide_no_nan(beta, layer_count))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(filters, grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.multiply(tf.constant(1.90885420e-02, dtype=tf.float32), beta), grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(grad, layer_count), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(tf.math.divide_no_nan(beta, layer_count))
```
- `multi_arch_update_phen_id_8859`
  - criterion: `fmnist_score` = `0.8363090753555298`
  - adjusted_score: `0.9759757369756699`
  - task_score_total: `0.9759757369756699`
  - overall_score: `1.098206064105034`; fitness: `-1.098206064105034`
  - setup/run/genetic/generation: `multi_arch_update` / `run_3` / `10121` / `437`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `cifar10=0.13966666162014008, fmnist=0.8363090753555298`
  - operation/source: `mutation` / `iteration_437.json`
  - smart_phenotype: `divide_no_nan(add(grad, add(multiply(constant(6.23645483e-01), grad), alpha)), negative(layer_count))`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(multiply(constant(6.23645483e-01), grad), alpha)
beta = beta - grad
weights = weights - divide_no_nan(add(beta, alpha), negative(layer_count))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(tf.math.multiply(tf.constant(6.23645483e-01, dtype=tf.float32), grad), alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: grad, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(tf.math.multiply(layer_num, tf.math.negative(tf.math.square(layer_num))), sigma), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.divide_no_nan(tf.math.add(beta, alpha), tf.math.negative(layer_count))
```
- `multi_arch_update_phen_id_24532`
  - criterion: `fmnist_score` = `0.8362363576889038`
  - adjusted_score: `1.3082363605499268`
  - task_score_total: `1.3082363605499268`
  - overall_score: `1.071597582101822`; fitness: `-1.071597582101822`
  - setup/run/genetic/generation: `multi_arch_update` / `run_0` / `732` / `65`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `cifar10=0.47200000286102295, fmnist=0.8362363576889038`
  - operation/source: `crossover+mutation` / `iteration_65.json`
  - smart_phenotype: `negative(divide_no_nan(subtract(grad, add(negative(sqrt(subtract(layer_count, multiply(layer_count, constant(9.96148968e-01))))), grad)), divide_no_nan(add(constant(9.99372875e-01), layer_num), grad)))`
  - advanced_readable_phenotype:

```text
alpha = alpha - negative(divide_no_nan(subtract(grad, add(negative(sqrt(subtract(layer_count, multiply(layer_count, constant(9.96148968e-01))))), grad)), divide_no_nan(add(constant(9.99372875e-01), layer_num), grad)))
weights = weights - alpha

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.negative(tf.math.divide_no_nan(tf.math.subtract(grad, tf.math.add(tf.math.negative(tf.math.sqrt(tf.math.subtract(layer_count, tf.math.multiply(layer_count, tf.constant(9.96148968e-01, dtype=tf.float32))))), grad)), tf.math.divide_no_nan(tf.math.add(tf.constant(9.99372875e-01, dtype=tf.float32), layer_num), grad))), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: strides, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: dilation_rate, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: alpha
```
- `multi_arch_update_phen_id_17836`
  - criterion: `fmnist_score` = `0.8361818194389343`
  - adjusted_score: `1.3568484783172607`
  - task_score_total: `1.3568484783172607`
  - overall_score: `1.0729006052017211`; fitness: `-1.0729006052017211`
  - setup/run/genetic/generation: `multi_arch_update` / `run_2` / `8482` / `497`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `cifar10=0.5206666588783264, fmnist=0.8361818194389343`
  - operation/source: `mutation` / `iteration_497.json`
  - smart_phenotype: `divide_no_nan(square(negative(divide_no_nan(add(grad, grad), add(layer_count, subtract(divide_no_nan(square(add(divide_no_nan(dilation_rate, square(constant(8.92947854e-01))), kernel_size)), add(pool_size, constant(5.75728612e-03))), grad))))), negative(divide_no_nan(add(grad, grad), add(layer_count, subtract(divide_no_nan(square(add(divide_no_nan(dilation_rate, square(constant(8.92947854e-01))), kernel_size)), add(pool_size, constant(5.75728612e-03))), grad)))))`
  - advanced_readable_phenotype:

```text
beta = beta - negative(divide_no_nan(add(grad, grad), add(layer_count, subtract(divide_no_nan(square(add(divide_no_nan(dilation_rate, square(constant(8.92947854e-01))), kernel_size)), add(pool_size, constant(5.75728612e-03))), grad))))
weights = weights - divide_no_nan(square(beta), beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: grad, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.negative(tf.math.divide_no_nan(tf.math.add(grad, grad), tf.math.add(layer_count, tf.math.subtract(tf.math.divide_no_nan(tf.math.square(tf.math.add(tf.math.divide_no_nan(dilation_rate, tf.math.square(tf.constant(8.92947854e-01, dtype=tf.float32))), kernel_size)), tf.math.add(pool_size, tf.constant(5.75728612e-03, dtype=tf.float32))), grad)))), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(tf.math.multiply(tf.math.square(layer_count), tf.math.pow(tf.math.add(layer_num, filters), grad)), sigma), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.divide_no_nan(tf.math.square(beta), beta)
```
- `multi_arch_update_phen_id_2733`
  - criterion: `fmnist_score` = `0.8360909223556519`
  - adjusted_score: `1.3200909197330475`
  - task_score_total: `1.3200909197330475`
  - overall_score: `1.083016973733902`; fitness: `-1.083016973733902`
  - setup/run/genetic/generation: `multi_arch_update` / `run_0` / `2695` / `323`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `cifar10=0.48399999737739563, fmnist=0.8360909223556519`
  - operation/source: `mutation` / `iteration_323.json`
  - smart_phenotype: `add(negative(divide_no_nan(subtract(grad, add(negative(sqrt(subtract(layer_count, multiply(layer_count, constant(9.96148968e-01))))), grad)), divide_no_nan(add(constant(6.69753570e-01), layer_num), grad))), pool_size)`
  - advanced_readable_phenotype:

```text
alpha = alpha - negative(divide_no_nan(subtract(grad, add(negative(sqrt(subtract(layer_count, multiply(layer_count, constant(9.96148968e-01))))), grad)), divide_no_nan(add(constant(6.69753570e-01), layer_num), grad)))
beta = beta - pool_size
weights = weights - add(alpha, beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.negative(tf.math.divide_no_nan(tf.math.subtract(grad, tf.math.add(tf.math.negative(tf.math.sqrt(tf.math.subtract(layer_count, tf.math.multiply(layer_count, tf.constant(9.96148968e-01, dtype=tf.float32))))), grad)), tf.math.divide_no_nan(tf.math.add(tf.constant(6.69753570e-01, dtype=tf.float32), layer_num), grad))), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: pool_size, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.constant(5.55606489e-05, dtype=tf.float32), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(alpha, beta)
```
- `multi_arch_update_phen_id_17971`
  - criterion: `fmnist_score` = `0.83478182554245`
  - adjusted_score: `1.3494485020637512`
  - task_score_total: `1.3494485020637512`
  - overall_score: `1.2343127131462097`; fitness: `-1.2343127131462097`
  - setup/run/genetic/generation: `multi_arch_update` / `run_2` / `6830` / `399`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `cifar10=0.5146666765213013, fmnist=0.83478182554245`
  - operation/source: `mutation` / `iteration_399.json`
  - smart_phenotype: `divide_no_nan(square(negative(divide_no_nan(add(grad, grad), add(layer_count, subtract(divide_no_nan(square(add(divide_no_nan(pool_size, square(constant(8.92947854e-01))), kernel_size)), add(pool_size, constant(5.75728612e-03))), grad))))), negative(divide_no_nan(add(grad, grad), add(layer_count, subtract(divide_no_nan(square(add(divide_no_nan(pool_size, square(constant(8.92947854e-01))), kernel_size)), add(pool_size, constant(5.75728612e-03))), grad)))))`
  - advanced_readable_phenotype:

```text
beta = beta - negative(divide_no_nan(add(grad, grad), add(layer_count, subtract(divide_no_nan(square(add(divide_no_nan(pool_size, square(constant(8.92947854e-01))), kernel_size)), add(pool_size, constant(5.75728612e-03))), grad))))
weights = weights - divide_no_nan(square(beta), beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: grad, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.negative(tf.math.divide_no_nan(tf.math.add(grad, grad), tf.math.add(layer_count, tf.math.subtract(tf.math.divide_no_nan(tf.math.square(tf.math.add(tf.math.divide_no_nan(pool_size, tf.math.square(tf.constant(8.92947854e-01, dtype=tf.float32))), kernel_size)), tf.math.add(pool_size, tf.constant(5.75728612e-03, dtype=tf.float32))), grad)))), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.multiply(tf.math.square(layer_num), tf.math.pow(tf.math.add(layer_count, padding), grad)), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.divide_no_nan(tf.math.square(beta), beta)
```
- `multi_arch_update_phen_id_7440`
  - criterion: `fmnist_score` = `0.83478182554245`
  - adjusted_score: `1.2661151587963104`
  - task_score_total: `1.2661151587963104`
  - overall_score: `1.2685757517814635`; fitness: `-1.2685757517814635`
  - setup/run/genetic/generation: `multi_arch_update` / `run_3` / `10206` / `441`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `cifar10=0.4313333332538605, fmnist=0.83478182554245`
  - operation/source: `mutation` / `iteration_441.json`
  - smart_phenotype: `divide_no_nan(add(add(multiply(strides, grad), alpha), grad), negative(add(multiply(negative(square(add(strides, layer_num))), constant(8.19823407e-01)), sigma)))`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(multiply(strides, grad), alpha)
beta = beta - grad
sigma = sigma - add(multiply(negative(square(add(strides, layer_num))), constant(8.19823407e-01)), sigma)
weights = weights - divide_no_nan(add(alpha, beta), negative(sigma))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(tf.math.multiply(strides, grad), alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: grad, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(tf.math.multiply(tf.math.negative(tf.math.square(tf.math.add(strides, layer_num))), tf.constant(8.19823407e-01, dtype=tf.float32)), sigma), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.divide_no_nan(tf.math.add(alpha, beta), tf.math.negative(sigma))
```
- `multi_arch_update_phen_id_22114`
  - criterion: `fmnist_score` = `0.8346182107925415`
  - adjusted_score: `1.339284896850586`
  - task_score_total: `1.339284896850586`
  - overall_score: `0.9428606152534484`; fitness: `-0.9428606152534484`
  - setup/run/genetic/generation: `multi_arch_update` / `run_2` / `6686` / `392`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `cifar10=0.5046666860580444, fmnist=0.8346182107925415`
  - operation/source: `crossover+mutation` / `iteration_392.json`
  - smart_phenotype: `negative(divide_no_nan(add(grad, grad), add(layer_num, subtract(divide_no_nan(square(add(divide_no_nan(padding, square(constant(9.49275639e-01))), strides)), add(pool_size, constant(6.27124987e-04))), grad))))`
  - advanced_readable_phenotype:

```text
beta = beta - negative(divide_no_nan(add(grad, grad), add(layer_num, subtract(divide_no_nan(square(add(divide_no_nan(padding, square(constant(9.49275639e-01))), strides)), add(pool_size, constant(6.27124987e-04))), grad))))
weights = weights - beta

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(alpha, alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.negative(tf.math.divide_no_nan(tf.math.add(grad, grad), tf.math.add(layer_num, tf.math.subtract(tf.math.divide_no_nan(tf.math.square(tf.math.add(tf.math.divide_no_nan(padding, tf.math.square(tf.constant(9.49275639e-01, dtype=tf.float32))), strides)), tf.math.add(pool_size, tf.constant(6.27124987e-04, dtype=tf.float32))), grad)))), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(layer_num, sigma), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: beta
```
- `multi_arch_update_phen_id_17945`

### Task: tiny_imagenet

- `multi_arch_update_phen_id_22729`
- `multi_arch_update_phen_id_22661`
- `multi_arch_update_phen_id_22702`
- `multi_arch_update_phen_id_22713`
- `multi_arch_update_phen_id_22676`
- `multi_arch_update_phen_id_22716`

## multi_no_arch_update

### Adjusted Overall

- `multi_no_arch_update_phen_id_45499`
- `multi_no_arch_update_phen_id_42625`
- `multi_no_arch_update_phen_id_42800`
- `multi_no_arch_update_phen_id_44343`
- `multi_no_arch_update_phen_id_44059`
- `multi_no_arch_update_phen_id_45488`
- `multi_no_arch_update_phen_id_44258`
- `multi_no_arch_update_phen_id_44220`
- `multi_no_arch_update_phen_id_44390`
- `multi_no_arch_update_phen_id_44212`
  - criterion: `adjusted_score` = `2.444897025823593`
  - adjusted_score: `2.444897025823593`
  - task_score_total: `2.444897025823593`
  - overall_score: `3.268133336305618`; fitness: `-3.268133336305618`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_2` / `1587` / `37`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.7603333592414856, cifar100=0.5786666870117188, fmnist=0.8325636386871338, tiny_imagenet=0.273333340883255`
  - operation/source: `mutation` / `iteration_37.json`
  - smart_phenotype: `multiply(negative(add(multiply(subtract(grad, beta), constant(1.27951705e-01)), grad)), sqrt(constant(1.90885420e-02)))`
  - advanced_readable_phenotype:

```text
beta = beta - negative(add(multiply(subtract(grad, beta), constant(1.27951705e-01)), grad))
weights = weights - multiply(beta, sqrt(constant(1.90885420e-02)))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: tf.math.divide_no_nan(tf.math.divide_no_nan(grad, tf.math.sqrt(grad)), tf.math.add(tf.math.add(tf.constant(9.58166060e-01, dtype=tf.float32), tf.math.sqrt(tf.math.add(tf.math.add(tf.math.subtract(tf.math.multiply(grad, grad), tf.math.divide_no_nan(tf.math.add(grad, alpha), grad)), alpha), tf.constant(9.99720385e-01, dtype=tf.float32)))), tf.math.sqrt(grad))), lambda shape, alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(grad, beta), tf.constant(1.27951705e-01, dtype=tf.float32)), grad)), lambda shape, alpha, beta, sigma, grad: tf.math.pow(tf.math.sqrt(tf.math.sqrt(tf.constant(2.11963334e-01, dtype=tf.float32))), grad), lambda shape, alpha, beta, sigma, grad: tf.math.multiply(beta, tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)))
```

### Task-Score Overall

- `multi_no_arch_update_phen_id_45499`
- `multi_no_arch_update_phen_id_42625`
- `multi_no_arch_update_phen_id_42800`
- `multi_no_arch_update_phen_id_44343`
- `multi_no_arch_update_phen_id_44059`
- `multi_no_arch_update_phen_id_45488`
- `multi_no_arch_update_phen_id_44258`
- `multi_no_arch_update_phen_id_44220`
- `multi_no_arch_update_phen_id_44390`
- `multi_no_arch_update_phen_id_44212`

### Task: cifar10

- `multi_no_arch_update_phen_id_42800`
- `multi_no_arch_update_phen_id_43849`
- `multi_no_arch_update_phen_id_45488`
- `multi_no_arch_update_phen_id_44390`
- `multi_no_arch_update_phen_id_44258`
- `multi_no_arch_update_phen_id_44144`
- `multi_no_arch_update_phen_id_44059`
- `multi_no_arch_update_phen_id_42625`
- `multi_no_arch_update_phen_id_44220`
- `multi_no_arch_update_phen_id_44078`

### Task: cifar100

- `multi_no_arch_update_phen_id_45499`
- `multi_no_arch_update_phen_id_44343`
- `multi_no_arch_update_phen_id_42800`
- `multi_no_arch_update_phen_id_44144`
- `multi_no_arch_update_phen_id_42625`
- `multi_no_arch_update_phen_id_44220`
- `multi_no_arch_update_phen_id_44258`
- `multi_no_arch_update_phen_id_44078`
- `multi_no_arch_update_phen_id_44059`
- `multi_no_arch_update_phen_id_44390`

### Task: fmnist

- `multi_no_arch_update_phen_id_35199`
- `multi_no_arch_update_phen_id_34826`
- `multi_no_arch_update_phen_id_44052`
- `multi_no_arch_update_phen_id_33247`
- `multi_no_arch_update_phen_id_33120`
- `multi_no_arch_update_phen_id_32961`
- `multi_no_arch_update_phen_id_34294`
- `multi_no_arch_update_phen_id_33446`
- `multi_no_arch_update_phen_id_33009`
- `multi_no_arch_update_phen_id_33124`

### Task: tiny_imagenet

- `multi_no_arch_update_phen_id_42625`
- `multi_no_arch_update_phen_id_45499`
- `multi_no_arch_update_phen_id_44343`
- `multi_no_arch_update_phen_id_44059`
- `multi_no_arch_update_phen_id_44212`
- `multi_no_arch_update_phen_id_44199`
  - criterion: `tiny_imagenet_score` = `0.2709999978542328`
  - adjusted_score: `2.4448303282260895`
  - task_score_total: `2.4448303282260895`
  - overall_score: `3.2661111056804657`; fitness: `-3.2661111056804657`
  - setup/run/genetic/generation: `multi_no_arch_update` / `run_2` / `1647` / `40`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `cifar10=0.7599999904632568, cifar100=0.5836666822433472, fmnist=0.8301636576652527, tiny_imagenet=0.2709999978542328`
  - operation/source: `mutation` / `iteration_40.json`
  - smart_phenotype: `multiply(negative(add(multiply(subtract(grad, beta), constant(1.07052146e-01)), grad)), sqrt(constant(1.90885420e-02)))`
  - advanced_readable_phenotype:

```text
beta = beta - negative(add(multiply(subtract(grad, beta), constant(1.07052146e-01)), grad))
weights = weights - multiply(beta, sqrt(constant(1.90885420e-02)))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: tf.constant(9.96148968e-01, dtype=tf.float32), lambda shape, alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(grad, beta), tf.constant(1.07052146e-01, dtype=tf.float32)), grad)), lambda shape, alpha, beta, sigma, grad: tf.math.pow(tf.constant(2.11963334e-01, dtype=tf.float32), tf.math.sqrt(tf.constant(8.59898661e-03, dtype=tf.float32))), lambda shape, alpha, beta, sigma, grad: tf.math.multiply(beta, tf.math.sqrt(tf.constant(1.90885420e-02, dtype=tf.float32)))
```
- `multi_no_arch_update_phen_id_44220`
- `multi_no_arch_update_phen_id_42800`
- `multi_no_arch_update_phen_id_45488`
- `multi_no_arch_update_phen_id_44258`

## no_multi_arch_update

### Adjusted Overall

- `no_multi_arch_update_phen_id_52872`
- `no_multi_arch_update_phen_id_52912`
- `no_multi_arch_update_phen_id_52868`
- `no_multi_arch_update_phen_id_52874`
- `no_multi_arch_update_phen_id_56942`
- `no_multi_arch_update_phen_id_52873`
  - criterion: `adjusted_score` = `2.421213330512047`
  - adjusted_score: `2.421213330512047`
  - task_score_total: `None`
  - overall_score: `0.2810833305120468`; fitness: `-0.2810833305120468`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_4` / `985` / `44`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `tiny_imagenet=0.2810833305120468`
  - operation/source: `crossover+mutation` / `iteration_44.json`
  - smart_phenotype: `divide_no_nan(add(negative(grad), beta), sqrt(layer_num))`
  - advanced_readable_phenotype:

```text
beta = beta - add(negative(grad), beta)
weights = weights - divide_no_nan(beta, sqrt(layer_num))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(tf.constant(9.99657809e-01, dtype=tf.float32), alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.negative(grad), beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(grad, sigma), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.divide_no_nan(beta, tf.math.sqrt(layer_num))
```
- `no_multi_arch_update_phen_id_57029`
  - criterion: `adjusted_score` = `2.420263330821991`
  - adjusted_score: `2.420263330821991`
  - task_score_total: `None`
  - overall_score: `0.280133330821991`; fitness: `-0.280133330821991`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_3` / `587` / `10`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `tiny_imagenet=0.280133330821991`
  - operation/source: `mutation` / `iteration_10.json`
  - smart_phenotype: `subtract(add(pool_size, pool_size), add(grad, alpha))`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(grad, alpha)
weights = weights - subtract(add(pool_size, pool_size), alpha)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(grad, alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.add(layer_num, grad), beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(tf.math.divide_no_nan(tf.math.subtract(tf.math.subtract(layer_count, beta), layer_num), tf.math.negative(grad)), sigma), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.subtract(tf.math.add(pool_size, pool_size), alpha)
```
- `no_multi_arch_update_phen_id_52512`
  - criterion: `adjusted_score` = `2.416930000667572`
  - adjusted_score: `2.416930000667572`
  - task_score_total: `None`
  - overall_score: `0.27680000066757204`; fitness: `-0.27680000066757204`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_3` / `696` / `12`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `tiny_imagenet=0.27680000066757204`
  - operation/source: `mutation` / `iteration_12.json`
  - smart_phenotype: `add(subtract(square(add(divide_no_nan(constant(2.11963334e-01), divide_no_nan(multiply(kernel_size, layer_count), grad)), alpha)), grad), sigma)`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(divide_no_nan(constant(2.11963334e-01), divide_no_nan(multiply(kernel_size, layer_count), grad)), alpha)
sigma = sigma - add(subtract(square(alpha), grad), sigma)
weights = weights - sigma

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(tf.math.divide_no_nan(tf.constant(2.11963334e-01, dtype=tf.float32), tf.math.divide_no_nan(tf.math.multiply(kernel_size, layer_count), grad)), alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.constant(9.89496155e-01, dtype=tf.float32), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(tf.math.subtract(tf.math.square(alpha), grad), sigma), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: sigma
```
- `no_multi_arch_update_phen_id_50682`
  - criterion: `adjusted_score` = `2.41466333735466`
  - adjusted_score: `2.41466333735466`
  - task_score_total: `None`
  - overall_score: `0.27453333735466`; fitness: `-0.27453333735466`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_3` / `763` / `14`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `tiny_imagenet=0.27453333735466`
  - operation/source: `mutation` / `iteration_14.json`
  - smart_phenotype: `add(add(subtract(square(pool_size), grad), sigma), pool_size)`
  - advanced_readable_phenotype:

```text
beta = beta - pool_size
sigma = sigma - add(subtract(square(beta), grad), sigma)
weights = weights - add(sigma, beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(tf.math.divide_no_nan(grad, tf.math.divide_no_nan(tf.math.multiply(tf.math.divide_no_nan(layer_num, dilation_rate), layer_count), tf.math.subtract(grad, tf.math.add(alpha, tf.math.sqrt(grad))))), alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: pool_size, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(tf.math.subtract(tf.math.square(beta), grad), sigma), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(sigma, beta)
```
- `no_multi_arch_update_phen_id_54211`
  - criterion: `adjusted_score` = `2.4141300009536746`
  - adjusted_score: `2.4141300009536746`
  - task_score_total: `None`
  - overall_score: `0.2740000009536743`; fitness: `-0.2740000009536743`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_1` / `762` / `55`
  - is_evaluated: `True`
  - has_architecture: `True`
  - task_scores: `tiny_imagenet=0.2740000009536743`
  - operation/source: `mutation` / `iteration_55.json`
  - smart_phenotype: `negative(add(subtract(grad, add(square(pool_size), alpha)), beta))`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(square(pool_size), alpha)
beta = beta - add(subtract(grad, alpha), beta)
weights = weights - negative(beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(tf.math.square(pool_size), alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.subtract(grad, alpha), beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: grad, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(beta)
```

### Task-Score Overall


### With Architecture

- `no_multi_arch_update_phen_id_52872`
- `no_multi_arch_update_phen_id_52912`
- `no_multi_arch_update_phen_id_52868`
- `no_multi_arch_update_phen_id_52874`
- `no_multi_arch_update_phen_id_56942`
- `no_multi_arch_update_phen_id_52873`
- `no_multi_arch_update_phen_id_57029`
- `no_multi_arch_update_phen_id_52512`
- `no_multi_arch_update_phen_id_50682`
- `no_multi_arch_update_phen_id_54211`

### Without Architecture

- `no_multi_arch_update_phen_id_56833`
  - criterion: `adjusted_score` = `2.4096633361625672`
  - adjusted_score: `2.4096633361625672`
  - task_score_total: `None`
  - overall_score: `0.26953333616256714`; fitness: `-0.26953333616256714`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_3` / `565` / `9`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.26953333616256714`
  - operation/source: `mutation` / `iteration_9.json`
  - smart_phenotype: `subtract(add(add(beta, beta), add(beta, beta)), add(grad, alpha))`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(grad, alpha)
beta = beta - add(beta, beta)
weights = weights - subtract(add(beta, beta), alpha)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(grad, alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(beta, beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(tf.math.divide_no_nan(tf.math.negative(tf.math.square(beta)), sigma), sigma), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.subtract(tf.math.add(beta, beta), alpha)
```
- `no_multi_arch_update_phen_id_51473`
  - criterion: `adjusted_score` = `2.406796662693024`
  - adjusted_score: `2.406796662693024`
  - task_score_total: `None`
  - overall_score: `0.2666666626930237`; fitness: `-0.2666666626930237`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_4` / `493` / `8`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.2666666626930237`
  - operation/source: `mutation` / `iteration_8.json`
  - smart_phenotype: `add(negative(grad), beta)`
  - advanced_readable_phenotype:

```text
beta = beta - add(negative(grad), beta)
weights = weights - beta

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: alpha, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.negative(grad), beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(grad, sigma), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: beta
```
- `no_multi_arch_update_phen_id_54013`
  - criterion: `adjusted_score` = `2.40199665892601`
  - adjusted_score: `2.40199665892601`
  - task_score_total: `None`
  - overall_score: `0.26186665892601013`; fitness: `-0.26186665892601013`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_1` / `103` / `1`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.26186665892601013`
  - operation/source: `mutation` / `iteration_1.json`
  - smart_phenotype: `negative(add(grad, beta))`
  - advanced_readable_phenotype:

```text
beta = beta - add(grad, beta)
weights = weights - negative(beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.square(alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(grad, beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.pow(tf.math.add(tf.math.add(tf.math.sqrt(tf.math.sqrt(tf.math.add(layer_count, tf.constant(4.74768924e-01, dtype=tf.float32)))), layer_num), tf.constant(9.98279874e-01, dtype=tf.float32)), tf.math.subtract(grad, tf.math.divide_no_nan(layer_num, grad))), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(beta)
```
- `no_multi_arch_update_phen_id_54015`
  - criterion: `adjusted_score` = `2.4015299962806704`
  - adjusted_score: `2.4015299962806704`
  - task_score_total: `None`
  - overall_score: `0.2613999962806702`; fitness: `-0.2613999962806702`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_1` / `670` / `40`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.2613999962806702`
  - operation/source: `mutation` / `iteration_40.json`
  - smart_phenotype: `negative(add(grad, sigma))`
  - advanced_readable_phenotype:

```text
sigma = sigma - add(grad, sigma)
weights = weights - negative(sigma)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.square(grad), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(grad, beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(grad, sigma), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(sigma)
```
- `no_multi_arch_update_phen_id_50551`
  - criterion: `adjusted_score` = `2.394463335161209`
  - adjusted_score: `2.394463335161209`
  - task_score_total: `None`
  - overall_score: `0.2543333351612091`; fitness: `-0.2543333351612091`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_2` / `858` / `26`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.2543333351612091`
  - operation/source: `mutation` / `iteration_26.json`
  - smart_phenotype: `add(add(grad, alpha), beta)`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(grad, alpha)
beta = beta - add(alpha, beta)
weights = weights - beta

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(grad, alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(alpha, beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: layer_num, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: beta
```
- `no_multi_arch_update_phen_id_52521`
  - criterion: `adjusted_score` = `2.3930633353042605`
  - adjusted_score: `2.3930633353042605`
  - task_score_total: `None`
  - overall_score: `0.25293333530426027`; fitness: `-0.25293333530426027`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_3` / `689` / `12`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.25293333530426027`
  - operation/source: `mutation` / `iteration_12.json`
  - smart_phenotype: `add(subtract(square(constant(7.67413430e-04)), grad), sigma)`
  - advanced_readable_phenotype:

```text
sigma = sigma - add(subtract(square(constant(7.67413430e-04)), grad), sigma)
weights = weights - sigma

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(tf.math.divide_no_nan(tf.constant(2.11963334e-01, dtype=tf.float32), tf.math.divide_no_nan(tf.math.multiply(tf.math.divide_no_nan(pool_size, layer_count), layer_count), tf.math.subtract(grad, tf.math.add(grad, tf.math.sqrt(layer_count))))), alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.constant(8.59898661e-03, dtype=tf.float32), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(tf.math.subtract(tf.math.square(tf.constant(7.67413430e-04, dtype=tf.float32)), grad), sigma), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: sigma
```
- `no_multi_arch_update_phen_id_54185`
  - criterion: `adjusted_score` = `2.377796661540667`
  - adjusted_score: `2.377796661540667`
  - task_score_total: `None`
  - overall_score: `0.2376666615406672`; fitness: `-0.2376666615406672`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_1` / `1057` / `97`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.2376666615406672`
  - operation/source: `mutation` / `iteration_97.json`
  - smart_phenotype: `negative(add(subtract(grad, add(grad, alpha)), beta))`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(grad, alpha)
beta = beta - add(subtract(grad, alpha), beta)
weights = weights - negative(beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(grad, alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.subtract(grad, alpha), beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(units, sigma), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.negative(beta)
```
- `no_multi_arch_update_phen_id_50406`
  - criterion: `adjusted_score` = `2.371463336234093`
  - adjusted_score: `2.371463336234093`
  - task_score_total: `None`
  - overall_score: `0.23133333623409272`; fitness: `-0.23133333623409272`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_2` / `966` / `30`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.23133333623409272`
  - operation/source: `mutation` / `iteration_30.json`
  - smart_phenotype: `add(add(add(grad, alpha), add(grad, alpha)), beta)`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(grad, alpha)
beta = beta - add(add(alpha, alpha), beta)
weights = weights - beta

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(grad, alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.add(alpha, alpha), beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: layer_num, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: beta
```
- `no_multi_arch_update_phen_id_52871`
  - criterion: `adjusted_score` = `2.31412999499321`
  - adjusted_score: `2.31412999499321`
  - task_score_total: `None`
  - overall_score: `0.17399999499320984`; fitness: `-0.17399999499320984`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_4` / `1066` / `68`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.17399999499320984`
  - operation/source: `mutation` / `iteration_68.json`
  - smart_phenotype: `divide_no_nan(add(negative(grad), beta), sqrt(constant(9.96851186e-01)))`
  - advanced_readable_phenotype:

```text
beta = beta - add(negative(grad), beta)
weights = weights - divide_no_nan(beta, sqrt(constant(9.96851186e-01)))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(grad, alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(tf.math.negative(grad), beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: padding, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.divide_no_nan(beta, tf.math.sqrt(tf.constant(9.96851186e-01, dtype=tf.float32)))
```
- `no_multi_arch_update_phen_id_52413`
  - criterion: `adjusted_score` = `2.300796659474373`
  - adjusted_score: `2.300796659474373`
  - task_score_total: `None`
  - overall_score: `0.16066665947437286`; fitness: `-0.16066665947437286`
  - setup/run/genetic/generation: `no_multi_arch_update` / `run_2` / `814` / `24`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.16066665947437286`
  - operation/source: `mutation` / `iteration_24.json`
  - smart_phenotype: `add(square(add(add(grad, alpha), beta)), add(add(grad, alpha), beta))`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(grad, alpha)
beta = beta - add(alpha, beta)
weights = weights - add(square(beta), beta)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.math.add(grad, alpha), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.math.add(alpha, beta), lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: grad, lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.add(tf.math.square(beta), beta)
```

### Task: tiny_imagenet

- `no_multi_arch_update_phen_id_52872`
- `no_multi_arch_update_phen_id_52912`
- `no_multi_arch_update_phen_id_52868`
- `no_multi_arch_update_phen_id_52874`
- `no_multi_arch_update_phen_id_56942`
- `no_multi_arch_update_phen_id_52873`
- `no_multi_arch_update_phen_id_57029`
- `no_multi_arch_update_phen_id_52512`
- `no_multi_arch_update_phen_id_50682`
- `no_multi_arch_update_phen_id_54211`

## no_multi_no_arch_update

### Adjusted Overall

- `no_multi_no_arch_update_phen_id_58316`
- `no_multi_no_arch_update_phen_id_57728`
- `no_multi_no_arch_update_phen_id_57634`
  - criterion: `adjusted_score` = `2.4201966630268097`
  - adjusted_score: `2.4201966630268097`
  - task_score_total: `None`
  - overall_score: `0.2800666630268097`; fitness: `-0.2800666630268097`
  - setup/run/genetic/generation: `no_multi_no_arch_update` / `run_3` / `257` / `4`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.2800666630268097`
  - operation/source: `mutation` / `iteration_4.json`
  - smart_phenotype: `add(add(grad, beta), sigma)`
  - advanced_readable_phenotype:

```text
beta = beta - add(grad, beta)
sigma = sigma - add(beta, sigma)
weights = weights - sigma

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: tf.math.add(grad, alpha), lambda shape, alpha, beta, grad: tf.math.add(grad, beta), lambda shape, alpha, beta, sigma, grad: tf.math.add(beta, sigma), lambda shape, alpha, beta, sigma, grad: sigma
```
- `no_multi_no_arch_update_phen_id_58141`
  - criterion: `adjusted_score` = `2.4168633328723907`
  - adjusted_score: `2.4168633328723907`
  - task_score_total: `None`
  - overall_score: `0.27673333287239077`; fitness: `-0.27673333287239077`
  - setup/run/genetic/generation: `no_multi_no_arch_update` / `run_2` / `280` / `4`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.27673333287239077`
  - operation/source: `mutation` / `iteration_4.json`
  - smart_phenotype: `add(negative(multiply(constant(6.69753570e-01), grad)), alpha)`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(negative(multiply(constant(6.69753570e-01), grad)), alpha)
weights = weights - alpha

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: tf.math.add(tf.math.negative(tf.math.multiply(tf.constant(6.69753570e-01, dtype=tf.float32), grad)), alpha), lambda shape, alpha, beta, grad: grad, lambda shape, alpha, beta, sigma, grad: tf.math.divide_no_nan(tf.math.square(tf.math.sqrt(tf.constant(9.99060945e-01, dtype=tf.float32))), tf.math.subtract(tf.math.multiply(tf.math.multiply(tf.math.square(grad), tf.math.divide_no_nan(grad, grad)), grad), tf.math.negative(tf.math.square(tf.math.add(grad, alpha))))), lambda shape, alpha, beta, sigma, grad: alpha
```
- `no_multi_no_arch_update_phen_id_57738`
  - criterion: `adjusted_score` = `2.4166633354473115`
  - adjusted_score: `2.4166633354473115`
  - task_score_total: `None`
  - overall_score: `0.2765333354473114`; fitness: `-0.2765333354473114`
  - setup/run/genetic/generation: `no_multi_no_arch_update` / `run_0` / `553` / `48`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.2765333354473114`
  - operation/source: `mutation` / `iteration_48.json`
  - smart_phenotype: `add(add(multiply(grad, divide_no_nan(grad, grad)), beta), sigma)`
  - advanced_readable_phenotype:

```text
beta = beta - add(multiply(grad, divide_no_nan(grad, grad)), beta)
sigma = sigma - add(beta, sigma)
weights = weights - sigma

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: grad, lambda shape, alpha, beta, grad: tf.math.add(tf.math.multiply(grad, tf.math.divide_no_nan(grad, grad)), beta), lambda shape, alpha, beta, sigma, grad: tf.math.add(beta, sigma), lambda shape, alpha, beta, sigma, grad: sigma
```
- `no_multi_no_arch_update_phen_id_58644`
  - criterion: `adjusted_score` = `2.416663329486847`
  - adjusted_score: `2.416663329486847`
  - task_score_total: `None`
  - overall_score: `0.27653332948684695`; fitness: `-0.27653332948684695`
  - setup/run/genetic/generation: `no_multi_no_arch_update` / `run_1` / `249` / `4`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.27653332948684695`
  - operation/source: `mutation` / `iteration_4.json`
  - smart_phenotype: `multiply(add(square(constant(9.99932002e-01)), alpha), add(grad, sigma))`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(square(constant(9.99932002e-01)), alpha)
sigma = sigma - add(grad, sigma)
weights = weights - multiply(alpha, sigma)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: tf.math.add(tf.math.square(tf.constant(9.99932002e-01, dtype=tf.float32)), alpha), lambda shape, alpha, beta, grad: tf.math.subtract(tf.math.subtract(tf.constant(1.56514861e-02, dtype=tf.float32), tf.math.add(tf.math.sqrt(grad), grad)), alpha), lambda shape, alpha, beta, sigma, grad: tf.math.add(grad, sigma), lambda shape, alpha, beta, sigma, grad: tf.math.multiply(alpha, sigma)
```
- `no_multi_no_arch_update_phen_id_58677`
  - criterion: `adjusted_score` = `2.4155966626453402`
  - adjusted_score: `2.4155966626453402`
  - task_score_total: `None`
  - overall_score: `0.27546666264534`; fitness: `-0.27546666264534`
  - setup/run/genetic/generation: `no_multi_no_arch_update` / `run_1` / `329` / `6`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.27546666264534`
  - operation/source: `mutation` / `iteration_6.json`
  - smart_phenotype: `multiply(add(subtract(constant(9.99932002e-01), constant(4.74768924e-01)), alpha), add(grad, sigma))`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(subtract(constant(9.99932002e-01), constant(4.74768924e-01)), alpha)
sigma = sigma - add(grad, sigma)
weights = weights - multiply(alpha, sigma)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: tf.math.add(tf.math.subtract(tf.constant(9.99932002e-01, dtype=tf.float32), tf.constant(4.74768924e-01, dtype=tf.float32)), alpha), lambda shape, alpha, beta, grad: tf.math.subtract(tf.math.subtract(tf.constant(1.56514861e-02, dtype=tf.float32), tf.math.add(tf.math.sqrt(grad), grad)), alpha), lambda shape, alpha, beta, sigma, grad: tf.math.add(grad, sigma), lambda shape, alpha, beta, sigma, grad: tf.math.multiply(alpha, sigma)
```
- `no_multi_no_arch_update_phen_id_58824`
  - criterion: `adjusted_score` = `2.4137300031232836`
  - adjusted_score: `2.4137300031232836`
  - task_score_total: `None`
  - overall_score: `0.27360000312328336`; fitness: `-0.27360000312328336`
  - setup/run/genetic/generation: `no_multi_no_arch_update` / `run_0` / `231` / `5`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.27360000312328336`
  - operation/source: `mutation` / `iteration_5.json`
  - smart_phenotype: `negative(negative(add(add(grad, beta), sigma)))`
  - advanced_readable_phenotype:

```text
beta = beta - add(grad, beta)
sigma = sigma - add(beta, sigma)
weights = weights - negative(negative(sigma))

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: tf.constant(1.72012560e-03, dtype=tf.float32), lambda shape, alpha, beta, grad: tf.math.add(grad, beta), lambda shape, alpha, beta, sigma, grad: tf.math.add(beta, sigma), lambda shape, alpha, beta, sigma, grad: tf.math.negative(tf.math.negative(sigma))
```
- `no_multi_no_arch_update_phen_id_57755`
  - criterion: `adjusted_score` = `2.413730000143051`
  - adjusted_score: `2.413730000143051`
  - task_score_total: `None`
  - overall_score: `0.27360000014305114`; fitness: `-0.27360000014305114`
  - setup/run/genetic/generation: `no_multi_no_arch_update` / `run_0` / `332` / `14`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.27360000014305114`
  - operation/source: `mutation` / `iteration_14.json`
  - smart_phenotype: `add(add(multiply(sqrt(constant(9.98279874e-01)), grad), beta), sigma)`
  - advanced_readable_phenotype:

```text
beta = beta - add(multiply(sqrt(constant(9.98279874e-01)), grad), beta)
sigma = sigma - add(beta, sigma)
weights = weights - sigma

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: tf.constant(9.76737464e-01, dtype=tf.float32), lambda shape, alpha, beta, grad: tf.math.add(tf.math.multiply(tf.math.sqrt(tf.constant(9.98279874e-01, dtype=tf.float32)), grad), beta), lambda shape, alpha, beta, sigma, grad: tf.math.add(beta, sigma), lambda shape, alpha, beta, sigma, grad: sigma
```
- `no_multi_no_arch_update_phen_id_58613`
  - criterion: `adjusted_score` = `2.4132633374977113`
  - adjusted_score: `2.4132633374977113`
  - task_score_total: `None`
  - overall_score: `0.2731333374977112`; fitness: `-0.2731333374977112`
  - setup/run/genetic/generation: `no_multi_no_arch_update` / `run_1` / `359` / `8`
  - is_evaluated: `True`
  - has_architecture: `False`
  - task_scores: `tiny_imagenet=0.2731333374977112`
  - operation/source: `mutation` / `iteration_8.json`
  - smart_phenotype: `multiply(add(constant(9.80911458e-01), alpha), add(grad, sigma))`
  - advanced_readable_phenotype:

```text
alpha = alpha - add(constant(9.80911458e-01), alpha)
sigma = sigma - add(grad, sigma)
weights = weights - multiply(alpha, sigma)

```
  - phenotype:

```python
alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: tf.math.add(tf.constant(9.80911458e-01, dtype=tf.float32), alpha), lambda shape, alpha, beta, grad: tf.math.subtract(tf.math.subtract(tf.constant(9.49275639e-01, dtype=tf.float32), tf.math.add(tf.math.sqrt(grad), grad)), alpha), lambda shape, alpha, beta, sigma, grad: tf.math.add(grad, sigma), lambda shape, alpha, beta, sigma, grad: tf.math.multiply(alpha, sigma)
```

### Task-Score Overall


### Task: tiny_imagenet

- `no_multi_no_arch_update_phen_id_58316`
- `no_multi_no_arch_update_phen_id_57728`
- `no_multi_no_arch_update_phen_id_57634`
- `no_multi_no_arch_update_phen_id_58141`
- `no_multi_no_arch_update_phen_id_57738`
- `no_multi_no_arch_update_phen_id_58644`
- `no_multi_no_arch_update_phen_id_58677`
- `no_multi_no_arch_update_phen_id_58824`
- `no_multi_no_arch_update_phen_id_57755`
- `no_multi_no_arch_update_phen_id_58613`

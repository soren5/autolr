# Top Optimizers Report

Top N: `3`

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

## Cross-Setup Task: cifar10

- `multi_arch_update_phen_id_22676`
  - criterion: `cifar10_score` = `0.8019999861717224`
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
  - criterion: `cifar10_score` = `0.7956666946411133`
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

## Cross-Setup Task: cifar100

- `multi_no_arch_update_phen_id_45499`
- `multi_no_arch_update_phen_id_44343`
  - criterion: `cifar100_score` = `0.6146666407585144`
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
- `multi_no_arch_update_phen_id_42800`
  - criterion: `cifar100_score` = `0.612666666507721`
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

## multi_arch_update

### Adjusted Overall

- `multi_arch_update_phen_id_22661`
- `multi_arch_update_phen_id_22676`
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

### Task-Score Overall

- `multi_arch_update_phen_id_22661`
- `multi_arch_update_phen_id_22676`
- `multi_arch_update_phen_id_22729`

### With Architecture

- `multi_arch_update_phen_id_22661`
- `multi_arch_update_phen_id_22676`
- `multi_arch_update_phen_id_22729`

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

### Task: cifar10

- `multi_arch_update_phen_id_22676`
- `multi_arch_update_phen_id_22713`
- `multi_arch_update_phen_id_22702`

### Task: cifar100

- `multi_arch_update_phen_id_22661`
- `multi_arch_update_phen_id_22676`
- `multi_arch_update_phen_id_22702`

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

### Task: tiny_imagenet

- `multi_arch_update_phen_id_22729`
- `multi_arch_update_phen_id_22661`
- `multi_arch_update_phen_id_22702`

## multi_no_arch_update

### Adjusted Overall

- `multi_no_arch_update_phen_id_45499`
- `multi_no_arch_update_phen_id_42625`
- `multi_no_arch_update_phen_id_42800`

### Task-Score Overall

- `multi_no_arch_update_phen_id_45499`
- `multi_no_arch_update_phen_id_42625`
- `multi_no_arch_update_phen_id_42800`

### Task: cifar10

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
  - criterion: `cifar10_score` = `0.7833333611488342`
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

### Task: cifar100

- `multi_no_arch_update_phen_id_45499`
- `multi_no_arch_update_phen_id_44343`
- `multi_no_arch_update_phen_id_42800`

### Task: fmnist

- `multi_no_arch_update_phen_id_35199`
- `multi_no_arch_update_phen_id_34826`
- `multi_no_arch_update_phen_id_44052`

### Task: tiny_imagenet

- `multi_no_arch_update_phen_id_42625`
- `multi_no_arch_update_phen_id_45499`
- `multi_no_arch_update_phen_id_44343`

## no_multi_arch_update

### Adjusted Overall

- `no_multi_arch_update_phen_id_52872`
- `no_multi_arch_update_phen_id_52912`
  - criterion: `adjusted_score` = `2.4335299955654146`
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
  - criterion: `adjusted_score` = `2.433263330345154`
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

### Task-Score Overall


### With Architecture

- `no_multi_arch_update_phen_id_52872`
- `no_multi_arch_update_phen_id_52912`
- `no_multi_arch_update_phen_id_52868`

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

### Task: tiny_imagenet

- `no_multi_arch_update_phen_id_52872`
- `no_multi_arch_update_phen_id_52912`
- `no_multi_arch_update_phen_id_52868`

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

### Task-Score Overall


### Task: tiny_imagenet

- `no_multi_no_arch_update_phen_id_58316`
- `no_multi_no_arch_update_phen_id_57728`
- `no_multi_no_arch_update_phen_id_57634`

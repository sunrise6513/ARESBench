<div align="center">

# ARESBench: Robustness Evaluation for Image Classification Based on [Paddle](https://www.paddlepaddle.org.cn/)

</div>

## Abstract
This repository contains code for evaluating the adversarial robustness of classification models based on Paddle. The robust models are derived from our benchmark, [ARESBench](https://link.springer.com/article/10.1007/s11263-024-02196-3). Additionally, we have ported the code for AutoAttack and PGD to the Paddle framework.

## Preparation
**Dataset**
- Support ImageNet datasets for evaluation. For custom datasets, users should define their `paddle.io.Dataset` class and corresponding `transform`.

**Classification Model**
- Prepare your own models or use our models of ARESBench-SwinB and ARESBench-SwinL.

## Getting Started
- Define custom `paddle.io.Dataset` and `transform` and replace the original ones in eval_paddle.py if a new dataset is evaluated.

- We provide a command line interface to run adversarial robustness evaluation.
  ```bash
  run_paddle.sh
  ```

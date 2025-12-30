# Configurations of Ablation Models

Use the following configurations to train the ablation models.
You can add/update these configurations to the [experiment configs](/configs/experiment/).
The configuration of the [traffic simulation policy](/configs/model/smart.yaml) and the [GMM-based ego policy](/configs/model/ego_gmm.yaml) are similar.
The [experiment config](/configs/experiment/clsft.yaml) just overrides the configs in the [model config](/configs/model/smart.yaml)

## Data Augmentation
- Trajectory perturbation of [SMART](https://arxiv.org/abs/2207.05844). (Open-loop)
  ```
  token_processor:
    agent_token_sampling:
      num_k: 5
      temp: 1.0
  training_loss:
    rollout_as_gt: true
  ```
- Noisy tokenization of [Trajeglish](https://arxiv.org/abs/2312.04535).
  - Default. (Open-loop)
    ```
    token_processor:
      agent_token_sampling:
        num_k: 5
        temp: 1.0
    ```
  - Sampled from uniform distribution. (Open-loop)
    ```
    token_processor:
      agent_token_sampling:
        num_k: 5
        temp: 1e5
    ```
  - Sampled from policy predicted probability. (Closed-loop)
    ```
    training_rollout_sampling:
      criterium: topk_dist_sampled_with_prob
      num_k: 5
      temp: 1.0
    ```

## Closed-Loop Supervised Fine-tuning with Top-K Sampling
- Top-5
  ```
  training_rollout_sampling:
    criterium: topk_prob
    num_k: 5
    temp: 1.0
  ```
- Top-5 + distance based filtering
  ```
  training_rollout_sampling:
    criterium: topk_prob
    num_k: 5
    temp: 1.0
  training_loss:
    gt_thresh_scale_length: 1.0
  ```
- Top-5 + distance based sampling with super low temperature is equivalent to CAT-5, and that's exactly how we implemented CAT-K rollout.
  ```
  training_rollout_sampling:
    criterium: topk_prob_sampled_with_dist
    num_k: 5
    temp: 1e-5
  ```

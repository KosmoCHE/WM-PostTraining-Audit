"""WM-OPSD custom reward: constant 0.

The learning signal in privileged self-distillation comes entirely from the
distillation loss (loss_mode="sdpo"), not from a scalar reward. verl's fit loop
unconditionally computes a reward, and the built-in default_compute_score raises
NotImplementedError on our data_source, so we supply this no-op scorer.

Wire via: custom_reward_function.path=<this file> custom_reward_function.name=compute_score
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return 0.0

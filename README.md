# RL algorithm testing - Beaumont, Roucka
This is a repository contains experiments on Reinforcement Learning algorithms.

The code is based on clones of
- [Ai Safety Gridworlds] (https://github.com/deepmind/ai-safety-gridworlds)
- [Gym AI Safety Gridworlds] (https://github.com/TheMrCodes/gym-ai-safety-gridworlds)
- [Attainable Utility Preservation] (https://github.com/alexander-turner/attainable-utility-preservation)

## Notes
Goal is to understand and implement the training of an agent, following the relative reachability coverage function to peanalize its impact on the environment. Implement the *starting state*, *inaction*, *stepwise baseline* - and its updated version suggested in [[1]](#1) - which we call *stepwise rollout baseline*.

Used papers for inspiration and understanding:

## Relative Reachability (RR)
- [Penalizing side effects using stepwise relative reachability](https://arxiv.org/pdf/1711.09883.pdf) [[1]](#1)
- [Measuring and avoiding side effects using relative reachability](https://www.researchgate.net/profile/Viktoriya-Krakovna/publication/325557348_Measuring_and_avoiding_side_effects_using_relative_reachability/links/5bb7e5eaa6fdcc9552d46b02/Measuring-and-avoiding-side-effects-using-relative-reachability.pdf) [[2]](#2)

## Auxiliar Utility Preservation (AUP)
- [Avoiding side effects in complex environments](https://proceedings.neurips.cc/paper/2020/hash/f50a6c02a3fc5a3a5d4d9391f05f3efc-Abstract.html) [[3]](#3)
- [Conservative agency via attainable utility preservation](https://dl.acm.org/doi/abs/10.1145/3375627.3375851?casa_token=TOj2-yjPZYEAAAAA:4BKWRa1IBaYiDlTDq_ykSQ48wH0sMMHfzWd_3nN4EF7fqKF9iS7XYAXVkJV_WIpWtoC9wpRiFy4lHw) [[4]](#4)

[[1]](#1).

## References
<a id="1">[1]</a> 
Krakovna, V., Orseau, L., Kumar, R., Martic, M. and Legg, S., 2018.
Penalizing side effects using stepwise relative reachability. 
arXiv preprint arXiv:1806.01186.
<a id="2">[2]</a> 
Krakovna, V., Orseau, L., Martic, M. and Legg, S., 2018.
Measuring and avoiding side effects using relative reachability.
arXiv preprint arXiv:1806.01186.
<a id="3">[3]</a> 
Turner, A., Ratzlaff, N. and Tadepalli, P., 2020.
Avoiding side effects in complex environments. 
Advances in Neural Information Processing Systems, 33, pp.21406-21415.
<a id="4">[4]</a> 
Turner, A.M., Hadfield-Menell, D. and Tadepalli, P., 2020, February.
Conservative agency via attainable utility preservation.
In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society (pp. 385-391).

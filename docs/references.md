# References and attribution

This page tracks the principal sources behind the educational explanations and implementation choices. Links point to DOI records, author repositories, or official project documentation. Publisher PDFs are not redistributed in this repository.

## Process design and flowsheet generation

1. Laura Stops, Roel Leenhouts, Qinghe Gao, and Artur M. Schweidtmann. “Flowsheet generation through hierarchical reinforcement learning and graph neural networks.” *AIChE Journal* 69(1), e17938 (2023; online 2022). [doi:10.1002/aic.17938](https://doi.org/10.1002/aic.17938).
2. Qinghe Gao and Artur M. Schweidtmann. “Deep reinforcement learning for process design: Review and perspective.” *Current Opinion in Chemical Engineering* 44, 101012 (2024). [doi:10.1016/j.coche.2024.101012](https://doi.org/10.1016/j.coche.2024.101012).
3. Qinghe Gao, Haoyu Yang, S. M. Shanbhag, and Artur M. Schweidtmann. “Transfer learning for process design with reinforcement learning.” [arXiv:2302.03375](https://arxiv.org/abs/2302.03375).

## Graph neural networks

4. Justin Gilmer et al. “Neural Message Passing for Quantum Chemistry.” *ICML* (2017). [PMLR](https://proceedings.mlr.press/v70/gilmer17a.html).
5. Thomas N. Kipf and Max Welling. “Semi-Supervised Classification with Graph Convolutional Networks.” *ICLR* (2017). [OpenReview](https://openreview.net/forum?id=SJU4ayYgl).
6. Petar Veličković et al. “Graph Attention Networks.” *ICLR* (2018). [OpenReview](https://openreview.net/forum?id=rJXMpikCZ).
7. Keyulu Xu et al. “How Powerful are Graph Neural Networks?” *ICLR* (2019). [OpenReview](https://openreview.net/forum?id=ryGs6iA5Km).

## Reinforcement learning and hybrid actions

8. Richard S. Sutton et al. “Policy Gradient Methods for Reinforcement Learning with Function Approximation.” *NeurIPS* (2000). [Proceedings](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation).
9. John Schulman et al. “High-Dimensional Continuous Control Using Generalized Advantage Estimation.” (2015). [arXiv:1506.02438](https://arxiv.org/abs/1506.02438).
10. John Schulman et al. “Proximal Policy Optimization Algorithms.” (2017). [arXiv:1707.06347](https://arxiv.org/abs/1707.06347).
11. Zhou Fan et al. “Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space.” *IJCAI* (2019). [doi:10.24963/ijcai.2019/316](https://doi.org/10.24963/ijcai.2019/316).
12. Richard S. Sutton and Andrew G. Barto. *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press (2018). [Author site](http://incompleteideas.net/book/the-book-2nd.html).

## Software documentation

- [PyTorch documentation](https://pytorch.org/docs/stable/)
- [PyTorch Geometric: creating message-passing networks](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html)
- [Gymnasium documentation](https://gymnasium.farama.org/)
- [uv documentation](https://docs.astral.sh/uv/)

The [GNN and hierarchical actor–critic tutorial](tutorials/gnn_hierarchical_actor_critic.md) gives a longer guided reading sequence. The [paper-to-prototype mapping](paper_mapping.md) distinguishes cited concepts from this repository’s implementation choices and results.

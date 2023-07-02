## Deep 2BSDE Methods for Stochastic Control Problems
Some machine learning algorithms for solving stochastic control problems, used in my thesis and [1]

Usage: CLI `python main.py -P Pow --primal`

# Dependencies
- `tensorflow == 1.13.2`
- `python == 3.7`

# Problems available
- `Pow`, `H`, `Log`, `Quad`, `Yaari` - Utility maximisation problem with power, non-Hara, Log, quadratic and Yaari utility respectively.
- `LQ` - Linear Quadratic Problem

Problems are configured in `config.py`.

# Methods available
- `--primal` Primal DC2BSDE Method
- `--dual` Dual DC2BSDE Method
- `--smp` DCSMP Method
- `--bruteprimal` My implementation of [2]
- `--brutedual` Solving the dual problem using [2]
- `--hybrid` My implementation of Hybrid-Now [3] 
- `--pde` For Pow problem, solving the HJB equation using [4], when optimal control is plugged into the PDE.

Methods are configured in `method.py`. Solutions are given in `analytical.py`.

# Additional parameters
- `--graph` Graphs a few paths of each process, compared to analytical solution if available.
- `--quiet` Disables logging
- `--sequential` Implements methods one by one, rather than all at once

# References
  
- [1] Davey, Ashley, and Harry Zheng. "Deep learning for constrained utility maximisation." Methodology and Computing in Applied Probability (2020): 1-32. [paper](https://link.springer.com/article/10.1007/s11009-021-09912-3)
- [2] Han, Jiequn. "Deep learning approximation for stochastic control problems." arXiv preprint arXiv:1611.07422 (2016). [paper](https://arxiv.org/pdf/1611.07422)
- [3] Deep neural networks algorithms for stochastic control
problems on finite horizon, Part 2: numerical applications. [paper](https://hal.science/hal-01949221v1/file/Deepconsto-Partie2_Final.pdf)
- [4] Beck, Christian, Weinan E, and Arnulf Jentzen. "Machine learning approximation algorithms for high-dimensional fully nonlinear partial differential equations and second-order backward stochastic differential equations." Journal of Nonlinear Science 29 (2019): 1563-1619. [paper](https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/article/10.1007/s00332-018-9525-3&casa_token=AXu4597_6bwAAAAA:zOewSvtGXnouvpgTOERp8PFpEeTckwWnY6Xnb4GaFnJHiY2qM1MQaDKSmiRNyvb1jWfyqejeoJkr8iO6)
  

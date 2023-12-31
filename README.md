## Tensor Network Message Passing Algorithms

This repository contains the code used to generate the results in [One-step replica symmetry breaking in the language of tensor networks](https://arxiv.org/abs/2306.15004)

## Install

Just clone this repo, optionally create an environment (`conda create -n tnmpa python=3.11`), and run `poetry install` from the project's directory `tensor-message-passing`
 
## Examples

In `tnmpa/examples` you can find a few jupyter notebooks that demonstrate how to use the code. The files `ksat_*` compute solutions of kSAT instances. The files `compare_*` compute some high-level comparisons between massage passing, exact and tensor network contractions. Finally, `quimb_bp.ipynb` showcases the power of vectorized belief propagation for *large* tensor networks.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

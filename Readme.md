## Recursive Uncertainty Model Identification (RUMI)


#### Introduction
The control of dynamic systems naturally deals with the handling of uncertainties. A widespread approach is to design a controller based on the nominal system dynamics and leverage its inherent robustness properties, e.g. using an LQR-Controller. However, this approach does not incorporate safety specifications such as control or state constraints in the design process. This issue is overcome by *Robust Control*. A key ingredient for a robust design is the specification of the worst-case system uncertainty. The RUMI algorithm presents an approach to identify these uncertainty specifications during system operation. It starts with a conservative initial guess and reduces its conservativeness while gathering data from the system.

The strenghts of RUMI are
* its ability to deal with arbitrary uncertainties (e.g. non-Gaussian uncertainty distributions)
* convergence guarantees do not require any statistical assumptions
* small computational requirements (only vector operations, no matrix multiplications or inversions) which enables the use
on embedded systems


#### How it works
The algorithm combines a normalized least mean squares algorithm with a quantile estimator for uncertainty quantification. It delivers an upper and lower bound which are allowed to depend on an arbitrary number of features. These can also be
states or inputs of the dynamic system. RUMI is based on a batch formulation, its final result are bounds that cover a certain proportion of the data in each batch. Details on the algorithm are under review in:

A. Wischnewski, J. Betz, and B. Lohmann, *Real-Time Learning of Non-Gaussian Uncertainty Models for Autonomous Racing*

Please cite this publication when using the software provided in this repository.

#### What to expect from this repository?
This repository mainly serves as a flexible benchmark for different uncertainty learning methods. It does not aim at highly efficient implementations and therefore the computation times can only serve as a rough estimate. All of
the algorithms are implemented using the same interface, which allows easy benchmarking. There are three implementations available (see `src` for details):
* Recursive Uncertainty Model Identification (RUMI)
* Gaussian Process Regression with hyperparameter training for uncertainty quantification (see scikit-learn [documentation](https://scikit-learn.org/stable/modules/gaussian_process.html))
* Bayesian Linear Regression with variance estimation (see chapter 7.6.3. in *Machine Learning -
A Probabilistic Persepctive* by Kevin P. Murphy)

The repository provides also all the experiments (`paper_results`) conducted for the paper and the corresponding plots (`paper_plots`).

#### Getting started
The repository was tested with Python 3.8.2.

* Setup a virtual environment
* Install the required packages `pip install -r requirements.txt`
* Run the example in 'example.py'

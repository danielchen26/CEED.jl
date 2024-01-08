
---

# **Active sampling-based CEEDesigns : generating the posterior distribution of molecular state in a constrained feature space, a step-by-step guide**

## **Introduction**

In data-driven decision-making, the estimation of the posterior distribution of a variable of interest, given a set of constraints and historical data, is a critical task. This document provides a structured and detailed guide on how to derive and understand this posterior distribution \( q(\vec{e}_s' | \vec{e}_s) \). The methodology incorporates constraints on feature values and target value ranges, using a variant of importance sampling tailored to specific domain needs. The approach leverages historical data and a similarity-based method for weighting, enabling the derivation of the posterior distribution within the constraints of specific feature values and target ranges.



## Deriving \( q(\vec{e}_{s'} | \vec{e}_s) \) within the active (constrained) sampling-based approach

### Step 1: Define the Feature and Target Constraints

- \( \mathcal{C}{e} \): Hard constraint set for the feature values of \( \vec{e}_s \).
- \( \mathcal{C}{y} \): Soft constraint set for the target values of \( y \).

### Step 2: Prior Distribution and Historical Data

- Let \( p(\vec{e}s) \) be the prior distribution over the states.
- Let \( H \) represent the historical data matrix consisting of pairs \( (\vec{e}_h, y_h) \), where \( \vec{e}_h \) are historical states and \( y_h \) are the corresponding outcomes.


### Step 3: Define the Similarity Measure

- Similarity Function: Define a similarity measure \( \text{sim}(\vec{e}_s, \vec{e}_h^s) \) that quantifies the resemblance between the test state \( \vec{e}_s \) and historical states \( \vec{e}_h^s \). 
  - In the historical dataset \( H \), each row vector is represented by \( \vec{e}_h\), which has the complete dimension of the feature state space. When given the particially known features state \( \vec{e}_s \), the similarity measure is then calculated as follows: \( \text{sim}(\vec{e}_s, \vec{e}_h^s) \), which only uses the known dimensions of the feature state space.


### Step 4: Compute Constrained Weights (active sampling constraints)

- Weight Calculation: We calculate weights for each historical data point, compute a weight that incorporates both the similarity measure and the constraints:

\[ w_h = \text{sim}(\vec{e}_s, \vec{e}_h^s) \cdot \mathbb{1}{\mathcal{C}{e}}(\vec{e}_h) \cdot \mathbb{1}{\mathcal{C}{y}}(y_h) \]

where \( \mathbb{1}{\mathcal{C}} \) is an indicator function that is 1 if the argument satisfies the constraint \( \mathcal{C} \) and 0 otherwise.

### Step 5: Normalizing Weights

- Normalize the weights to ensure they sum to 1:

\[ w_h' = \frac{w_h}{\sum_{h \in H} w_h} \]

### Step 6: Define the Sampling Distribution

- Define a sampling distribution \( p(\vec{e}_{s'}|H) \) that samples states \( \vec{e}_{s'} \) from the historical data weighted by the normalized weights \( w_{h}' \).
- The sampled state \( \vec{e}_{s'} \) is considered as a new observation, which augments the current \( \vec(e}_{s} \) with additional dimensions by appending sampled data vector \( \vec{e}_{s}^* \) from the process. 

### Step 7: Compute the Posterior Distribution

- The posterior distribution \( q(\vec{e}_{s'} | \vec{e}_s) \) over the states, given the test state \( \vec{e}_s \), is estimated by the weighted samples:

\[ q(\vec{e}_{s'} | \vec{e}_s) = \sum_{h \in H} w_{h}' \cdot \delta(\vec{e}_{s'} - \vec{e}_h^{s'}) \]

where \( \delta \) is the Dirac delta function.

### Step 8: Iteratively compute posterior within the CEED MDP framework
- At each step, the posterior distribution \( q(\vec{e}_{s'} | \vec{e}_s) \) is updated to reflect the new weights, taking into account the scenarios of active sampling. This updated posterior is essential for the CEED MDP framework to compute the optimal state-action pair that is required for the multi-objective optimization process.

### Final Notes

- The derived posterior \( q(\vec{e}_{s'} | \vec{e}_s) \) is an approximation, as it relies on the historical data and the constraints applied.
- The constraints \( \mathcal{C}_{e} \) and \( \mathcal{C}_{y} \) play a crucial role in shaping the posterior distribution by limiting the influence of the historical data to only those points that satisfy the constraints.
- This process assumes that the constraints and similarity measure are well-defined and relevant to the problem at hand.

This derivation provides a structured approach to estimate a constrained posterior distribution in a problem where constraints on features and target values are critical. It allows for actively incorporating new observations into the posterior estimate, making it dynamic and adaptive to new information.

This methodology provides a practical framework with rigorous derivation for estimating the posterior distribution in a constrained feature space. By iteratively updating the distribution based on new data and predefined constraints, it enables more focused and relevant decision-making. It's particularly useful in scenarios where specific regions of the feature space are of higher interest or relevance.












# ======== Review of traditional importance sampling
Importance Sampling is a technique used to estimate properties of a particular distribution, while only having samples generated from a different distribution than the one of interest. It's particularly useful when it's difficult to sample from the distribution of interest, but you have another distribution that is easier to sample from and is somewhat similar to your target distribution.

Here's how it works, step by step:

### Step 1: Identify the target distribution
Suppose you want to estimate an expectation of a function under a target probability distribution \( p(x) \), which is difficult to sample from directly. The expectation is defined as:

\[ E_{p}[f(x)] = \int f(x)p(x)dx \]

### Step 2: Choose a proposal distribution
Select a proposal distribution \( q(x) \) from which it is easy to sample. The proposal distribution should be non-zero wherever the target distribution is non-zero (i.e., \( q(x) > 0 \) whenever \( p(x) > 0 \)).

### Step 3: Generate samples
Draw \( N \) samples \( \{x_1, x_2, ..., x_N\} \) from the proposal distribution \( q(x) \).

### Step 4: Compute weights
For each sample \( x_i \), compute a weight \( w_i \) which is the ratio of the probabilities of \( x_i \) under the two distributions:

\[ w_i = \frac{p(x_i)}{q(x_i)} \]

These weights adjust for the difference in the probability of the samples under the two distributions.

### Step 5: Estimate the expectation
Calculate the weighted average of the function \( f(x) \) using the samples and their weights:

\[ \hat{E}_{p}[f(x)] = \frac{1}{N} \sum_{i=1}^{N} w_i f(x_i) \]

### Step 6: Normalize the weights (optional)
Sometimes the weights are normalized to make them sum to 1, which can help with numerical stability:

\[ \hat{E}_{p}[f(x)] = \sum_{i=1}^{N} \left( \frac{w_i}{\sum_{j=1}^{N}w_j} \right) f(x_i) \]


# Difference between CEEDesigns.jl active sampling vs traditional importance sampling.
In the current active sampling approach, we are extending the traditional importance sampling method to better suit the needs of our specific problem domain. Traditional importance sampling is a statistical technique used to **estimate properties of a particular distribution**, while having only samples generated from a different distribution rather than the distribution of interest. The __key idea__ is to draw samples from an easy-to-sample proposal distribution and then reweight those samples by the ratio of the target distribution to the proposal distribution.

However, in our active sampling approach, we are not just interested in estimating properties of a distribution, but we also want to focus our sampling on specific regions of the feature and target spaces that are of higher interest or relevance. To achieve this, we introduce constraints on feature values and target value ranges, which are specified as desirable ranges. These constraints are incorporated into the calculation of the weights used in importance sampling.

In addition to the similarity measure used in traditional importance sampling, our weights are also influenced by whether the historical states fall within the **desirable range (hard constraint)** and by the **importance of the data point (soft constraint)**. This allows us to adjust the sampling process to focus more on the regions of the feature and target spaces that we are most interested in.

Our active sampling approach provides a practical and rigorously derived framework for estimating the posterior distribution in a constrained feature space. It uniquely incorporates data augmentation, actively sampling and adding new observations to the dataset with each iteration. As the posterior distribution is updated with each new observation, the process refines the estimation of the distribution and adapts to new data and state augmentation. This approach respects predefined constraints, ensuring that the iterative updates focus on relevant regions of the feature space. This makes it particularly useful in scenarios where data is collected sequentially over time for decision making, and where specific regions of the feature space are of higher interest or relevance.
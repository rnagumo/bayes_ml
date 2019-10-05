
module BayesNeuralNet

"""
Bayesian Neural Network (3 layer Perceptron)
"""

using Distributions

struct BayesNN
    M::Int  # Dimension of input
    K::Int  # Dimension of latent
    D::Int  # Dimension of output
    sigma2_w::Float64  # variance of w
    sigma2_y::Float64  # variance of y
end

# -----------------------------------------------------------
# Utils
# -----------------------------------------------------------

function basis_func(x::Array{Float64, 1}, D::Int)
    X = repeat(x, 1, D) .^ (0:D - 1)'
    return X
end

# -----------------------------------------------------------
# Sampling Functions
# -----------------------------------------------------------

function sample_fn(model::BayesNN, x::Array{Float64, 2})
    """
    Sample function given data x
    """
end

end

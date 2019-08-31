
module PoissonHMM

"""
Variational Inference for Bayesian NMF

p(X, S, lambda, pi, A) 
= p(X|S, lambda) p(S|pi, A) p(lambda) p(pi) p(A)
= p(lambda) p(pi) p(A) p(x_1|s_1, lambda) p(s_1|pi)
      prod_{n=2}^{N} p(x_n|s_n, lambda) p(s_n|s_{n-1}, A)

p(lambda_k) = Gam(lambda_k|a, b)
p(pi) = Dir(pi|alpha)
p(A_{:, i}) = Dir(A_{:, i}|beta_{:, i})
p(s_1|pi) = Cat(s_1|pi)
p(x_n|s_n, lambda) = Poi(x_n|lambda_k)^s_n,k
p(s_n|s_n-1, A) = prod_{i=1}^{K} Cat(s_n|A_{:, i})^{s_{n-1, i}}
"""

using LinearAlgebra
using Distributions
using StatsFuns
using SpecialFunctions

export PoissonHMMModel, VI, sample_data

struct PoissonHMMModel
    K::Int
    a::Array{Float64, 1}  # K
    b::Array{Float64, 1}  # K
    alpha::Array{Float64, 1}  # K
    beta::Array{Float64, 2}  # K*K
end

# -----------------------------------------------------------
# Utils
# -----------------------------------------------------------

eye(D::Int) = Matrix{Float64}(I, D, D)

function sqsum(mat::Array{Float64}, idx::Int)
    return dropdims(sum(mat, dims=idx), dims=idx)
end

# -----------------------------------------------------------
# Functions
# -----------------------------------------------------------

function sample_data(N::Int, model::PoissonHMMModel)
    """
    Sample parameters and data
    """

    # Dimension
    K = model.K

    # Sample lambda
    lambda = zeros(K)
    for k in 1:K
        lambda[k] = rand(Gamma(model.a[k], 1.0 / model.b[k]))
    end

    # Sample pi
    pi = rand(Dirichlet(model.alpha))

    # Sample A
    A = zeros(K, K)
    for k in 1:K
        A[:, k] = rand(Dirichlet(model.beta[:, k]))
    end

    # Sample s (latent variable)
    S = zeros(K, N)


    # Sample x
    X = zeros(N)

    return X, S, A, pi, lambda
end

function init_latent_variable(X::Array{Float64, 2}, prior::PoissonHMMModel)
    # Dimension
    K = prior.K

end

function VI(X::Array{Float64, 1}, prior::PoissonHMMModel, max_iter::Int)
    """
    Variational Inference for NMF
    """
    # Initialize latent variable
    # S = init_latent_variable(X, prior)
    S = 1
    posterior = deepcopy(prior)

    # Inference
    for iter in 1:max_iter
        
    end

    return posterior, S
end

end

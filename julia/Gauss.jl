
module Gauss

"""
Gaussian distribution 

Inference unknown mu and precision
"""

using LinearAlgebra
using Distributions
using StatsFuns
using SpecialFunctions

struct GaussModel
    D::Int
    m::Array{Float64, 1}  # D
    beta::Float64
    nu::Int64
    W::Array{Float64, 2}  # D*D
end

struct SampledGaussModel
    D::Int
    mu::Array{Float64, 1}
    Lambda::Array{Float64, 2}
end

struct SampledTDistModel
    D::Int
    nu::Int
    mu::Array{Float64, 1}
    Lambda::Array{Float64, 2}
end

# -----------------------------------------------------------
# Utils
# -----------------------------------------------------------

eye(D::Int) = Matrix{Float64}(I, D, D)

function sqsum(mat::Array{Float64}, idx::Int)
    return dropdims(sum(mat, dims=idx), dims=idx)
end

# -----------------------------------------------------------
# Sampling Functions
# -----------------------------------------------------------

function sample_model(model::GaussModel)
    """
    Sample parameters
    """

    Lambda = rand(Wishart(model.nu, model.W))
    mu = rand(MvNormal(model.m, inv(model.beta .* Lambda)))

    return SampledGaussModel(model.D, mu, Lambda)
end

function sample_data(N::Int, model::SampledGaussModel)
    """
    Sample data
    """

    # Dimension
    D = model.D

    # Sample data
    X = zeros(N, D)
    for n in 1:N
        X[n, :] = rand(MvNormal(model.mu, inv(model.Lambda)))
    end

    return X
end

function sample_prediction(N::Int, model::GaussModel)
    """
    Sample predictions from predictive distribution

    St(x* | nu, mu, Lambda)
    nu: degree of freedom
    mu: location
    Lambda: scale
    """

    # Dimension
    D = model.D

    # Parameters
    nu = 1 - D + model.nu
    mu = model.m
    Lambda = Matrix(Symmetric(
        ((1 - D + model.nu) * model.beta / (1 + model.beta)) .* model.W))

    # Sample
    X_pred = zeros(N, D)
    for n in 1:N
        X_pred[n, :] = rand(MvTDist(nu, mu, inv(Lambda)))
    end

    return X_pred, SampledTDistModel(D, nu, mu, Lambda)
end

# -----------------------------------------------------------
# Inference Functions
# -----------------------------------------------------------

function inference(X::Array{Float64, 2}, prior::GaussModel)

    # Dimension
    N, D = size(X)

    # Inference
    beta = N + prior.beta
    m = (sqsum(X, 1) + prior.beta .* prior.m) ./ beta
    nu = N + prior.nu
    W = inv(X' * X + prior.beta .* prior.m * prior.m'
            - beta .* m * m' + inv(prior.W))

    return GaussModel(D, m, beta, nu, W)
end

function variational_inference(p::SampledGaussModel)
    """
    Variational Inference for 2D Gauss

    Parameters
    ----------
    p : True 2D Gaussian

    Returns
    -------
    q_klqp : argmin KL(q||p)

    q_klpq : argmin KL(p||q)
    """

    # KL(q||p)
    q_klqp = SampledGaussModel(p.D, p.mu, p.Lambda .* eye(p.D))

    # KL(p||q)
    lambda_1 = (p.Lambda[1, 1] 
                - p.Lambda[1, 2] * inv(p.Lambda[2, 2]) * p.Lambda[2, 1])
    lambda_2 = (p.Lambda[2, 2]
                - p.Lambda[2, 1] * inv(p.Lambda[1, 1]) * p.Lambda[1, 2])
    q_klpq = SampledGaussModel(p.D, p.mu, [lambda_1 0; 0 lambda_2])

    return q_klqp, q_klpq
end

end

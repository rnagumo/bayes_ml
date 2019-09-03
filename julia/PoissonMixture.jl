
module PoissonMixture

"""
Poisson Mixture Model

p(x, z, lmd, pi) = p(x|z, lmd)p(z|pi)p(lmd)p(pi)

p(pi) = Dir(pi|alpha)
p(lmd) = Gam(lmd_d,k|a, b)
p(z|pi) = Cat(z_n|pi)
p(x|z, lmd) = Poi(x_n,d|lmd_d,k)^(z_n,k)
"""

using LinearAlgebra
using Distributions
using StatsFuns
using SpecialFunctions

export PoissonMixtureModel, sample_model, sample_data, VI, GS

struct PoissonMixtureModel
    D::Int
    K::Int
    alpha::Array{Float64, 1}  # K
    a::Array{Float64, 2}  # D*K
    b::Array{Float64, 1}  # K
end

struct SampledPoissonMixtureModel
    D::Int
    K::Int
    phi::Array{Float64, 1}  # K
    lambda::Array{Float64, 2}  # D*K
end

# -----------------------------------------------------------
# Utils
# -----------------------------------------------------------

eye(D::Int) = Matrix{Float64}(I, D, D)

function sqsum(mat::Array{Float64}, idx::Int)
    return dropdims(sum(mat, dims=idx), dims=idx)
end

function categorical_sample(p::Array{Float64}, N::Int64=1)
    """
    Convert sampled index to one-hot vector
    """
    K = length(p)
    S = zeros(N, K)
    for n in 1:N
        S[n, rand(Categorical(p))] = 1
    end

    if N == 1
        S = S[1, :]
    end

    return S
end

# -----------------------------------------------------------
# Sampling Functions
# -----------------------------------------------------------

function sample_model(model::PoissonMixtureModel)
    """
    Sample paramters
    """
    # Dimension
    D = model.D
    K = model.K

    # Sample phi
    phi = rand(Dirichlet(model.alpha))

    # Sample lambda
    lambda = zeros(D, K)
    for d in 1:D
        for k in 1:K
            lambda[d, k] = rand(Gamma(model.a[d, k], 1.0 / model.b[k]))
        end
    end

    return SampledPoissonMixtureModel(D, K, phi, lambda)
end

function sample_data(N::Int, model::SampledPoissonMixtureModel)
    """
    Sample data
    """
    # Dimension
    D = model.D
    K = model.K

    # Sample S (latent variable)
    S = zeros(N, K)
    for n in 1:N
        S[n, :] = categorical_sample(model.phi)
    end

    # Sample X (data)
    X = zeros(N, D)
    for n in 1:N
        for d in 1:D
            X[n, d] = rand(Poisson(model.lambda[d, argmax(S[n, :])]))
        end
    end

    return X, S
end

# -----------------------------------------------------------
# Variational Inference Functions
# -----------------------------------------------------------

function init_latent_variable(X::Array{Float64}, prior::PoissonMixtureModel)

    # Dimension
    N = size(X, 1)
    K = prior.K
    S = categorical_sample(ones(K) ./ K, N)
    return S
end

function update_S(X::Array{Float64}, posterior::PoissonMixtureModel)

    # Dimension
    N = size(X, 1)
    K = posterior.K

    # Expectations
    lmd_expt = posterior.a ./ posterior.b'
    ln_lmd_expt = digamma.(posterior.a) .- log.(posterior.b)'
    ln_phi_expt = digamma.(posterior.alpha) .- digamma(sum(posterior.alpha))

    # Update S
    ln_S_expt = zeros(N, K)
    for n in 1:N
        for k in 1:K
            ln_S_expt[n, k] = (X[n, :]' * ln_lmd_expt[:, k]
                                  - sum(lmd_expt[:, k])
                                  + ln_phi_expt[k])
        end
        ln_S_expt[n, :] .-= logsumexp(ln_S_expt[n, :])
    end

    return exp.(ln_S_expt)
end

function update_posterior(S::Array{Float64, 2}, X::Array{Float64},
                           prior::PoissonMixtureModel)

    alpha = sqsum(S, 1) + prior.alpha
    a = X' * S + prior.a
    b = sqsum(S, 1) + prior.b

    return PoissonMixtureModel(prior.D, prior.K, alpha, a, b)
end

function VI(X::Array{Float64}, prior::PoissonMixtureModel, max_iter::Int)
    """
    Variational Inference for PoissonMixtureModel
    """
    # Initialization
    S_expt = init_latent_variable(X, prior)
    posterior = update_posterior(S_expt, X, prior)

    # Inference
    for iter in 1:max_iter
        # E-step
        S_expt = update_S(X, posterior)

        # M-step
        posterior = update_posterior(S_expt, X, prior)
    end

    return posterior, S_expt
end

# -----------------------------------------------------------
# Gibbs Sampling Functions
# -----------------------------------------------------------

function sample_S(X::Array{Float64}, posterior::PoissonMixtureModel)

    # Dimension
    N = size(X, 1)
    K = posterior.K

    # Sample parameters
    model = sample_model(posterior)

    # Sample S
    ln_S = zeros(N, K)
    for n in 1:N
        for k in 1:K
            ln_S[n, k] = (X[n, :]' * log.(model.lambda[:, k])
                              - sum(model.lambda[:, k])
                              + log.(model.phi[k]))
        end
        ln_S[n, :] .-= logsumexp(ln_S[n, :])
    end

    return exp.(ln_S)
end

function GS(X::Array{Float64}, prior::PoissonMixtureModel, max_iter::Int)
    """
    Gibbs sampling for PoissonMixtureModel
    """
    # Initialization
    S = init_latent_variable(X, prior)
    posterior = update_posterior(S, X, prior)

    # Inference
    for iter in 1:max_iter
        # Sample latent variable
        S = sample_S(X, posterior)

        # update posterior
        posterior = update_posterior(S, X, prior)
    end

    return posterior, S
end

end

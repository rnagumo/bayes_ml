
module GaussianMixture

"""
Gaussian Mixture Model

p(x, z, mu, lambda, pi) = p(x|z, mu, lambda) p(s|pi) p(mu, lambda) p(pi)

p(x|z, mu, lambda) = prod^{K}_{k=1} N(x_n|mu_k, lambda_k^-1)^(s_{n,k})
p(s|pi) = Cat(s_n|pi)
p(mu_k, lambda_k) = N(mu_k|m, (beta*lambda_k)^-1) W(lambda_k|nu, W)
p(pi) = Dir(pi|alpha)
"""

using LinearAlgebra
using Distributions
using StatsFuns
using SpecialFunctions

export GaussianMixtureModel, sample_model, sample_data
export variational_inference

struct GaussianMixtureModel
    D::Int
    K::Int
    alpha::Array{Float64, 1}  # K
    m::Array{Float64, 2}  # K*D
    beta::Array{Float64, 1}  # K, positive
    nu::Array{Float64, 1}  # K, larger than D-1
    W::Array{Float64, 3}  # D*D*K
end

struct SampledGaussianMixtureModel
    D::Int
    K::Int
    phi::Array{Float64, 1}  # K
    mu::Array{Float64, 2}  # K*D
    Lambda::Array{Float64, 3}  # D*D*K
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

function sample_model(model::GaussianMixtureModel)
    # Dimension
    D = model.D
    K = model.K

    # Sample phi
    phi = rand(Dirichlet(model.alpha))

    # Sample Lambda
    Lambda = zeros(D, D, K)
    for k in 1:K
        Lambda[:, :, k] = rand(Wishart(model.nu[k], model.W[:, :, k]))
    end

    # Sample mu
    mu = zeros(K, D)
    for k in 1:K
        mu[k, :] = rand(MvNormal(model.m[k, :], 
                                 inv(model.beta[k] .* Lambda[:, :, k])))
    end

    return SampledGaussianMixtureModel(D, K, phi, mu, Lambda)
end

function sample_data(N::Int, model::SampledGaussianMixtureModel)
    # Dimension
    D = model.D
    K = model.K

    # Sample Z
    Z = zeros(N, K)
    for n in 1:N
        Z[n, :] = categorical_sample(model.phi)
    end

    # Sample X
    X = zeros(N, D)
    for n in 1:N
        k = argmax(Z[n, :])
        X[n, :] = rand(MvNormal(model.mu[k, :], inv(model.Lambda[:, :, k])))
    end

    return X, Z
end

# -----------------------------------------------------------
# Variational Inference Functions
# -----------------------------------------------------------

function init_latent_variable(N::Int, K::Int)
    Z = categorical_sample(ones(K) ./ K, N)

    return Z
end

function update_z(X::Array{Float64}, posterior::GaussianMixtureModel)
    # Dimension
    N, D = size(X)
    K = posterior.K

    # Expectations
    lmd_expt = zeros(D, D, K)
    for k in 1:K
        lmd_expt[:, :, k] = posterior.nu[k] .* posterior.W[:, :, k]
    end

    ln_lmd_det_expt = zeros(K)
    for k in 1:K
        for d in 1:D
            ln_lmd_det_expt[k] += digamma((posterior.nu[k] + 1 - d) / 2)
        end
        ln_lmd_det_expt[k] += D * log(2) + logdet(posterior.W[:, :, k])
    end

    lmd_mu_expt = zeros(K, D)
    for k in 1:K
        lmd_mu_expt[k, :] = (posterior.nu[k] .* posterior.W[:, :, k] 
                                * posterior.m[k, :])
    end

    mu_lmd_mu_expt = zeros(K)
    for k in 1:K
        mu_lmd_mu_expt[k] = ((posterior.nu[k] .* posterior.m[k, :]'
                             * posterior.W[:, :, k] * posterior.m[k, :])[1]
                             + D / posterior.beta[k])
    end

    ln_pi_expt = zeros(K)
    for k in 1:K
        ln_pi_expt[k] = (digamma(posterior.alpha[k]) 
                             - digamma(sum(posterior.alpha)))
    end

    # Update Z
    ln_Z_expt = zeros(N, K)
    for n in 1:N
        for k in 1:K
            ln_Z_expt[n, k] = (-0.5 * X[n, :]' * lmd_expt[:, :, k] * X[n, :]
                               + X[n, :]' * lmd_mu_expt[k, :]
                               - 0.5 * mu_lmd_mu_expt[k]
                               + 0.5 * ln_lmd_det_expt[k]
                               + ln_pi_expt[k])
        end
        ln_Z_expt[n, :] .-= logsumexp(ln_Z_expt[n, :])
    end

    return exp.(ln_Z_expt)
end

function update_posterior(Z::Array{Float64}, X::Array{Float64},
                          prior::GaussianMixtureModel)

    N, D = size(X)
    K = prior.K

    # Update parameters
    alpha = sqsum(Z, 1) + prior.alpha

    beta = sqsum(Z, 1) + prior.beta
    m = (Z' * X + repeat(prior.beta, 1, D) .* prior.m) ./ repeat(beta, 1, D)

    nu = sqsum(Z, 1) + prior.nu
    W = zeros(D, D, K)
    for k in 1:K
        W[:, :, k] = inv(X' * diagm(0 => Z[:, k]) * X
                         + prior.beta[k] .* prior.m[k, :] * prior.m[k, :]'
                         - beta[k] .* m[k, :] * m[k, :]'
                         + inv(prior.W[:, :, k]))
    end

    return GaussianMixtureModel(D, K, alpha, m, beta, nu, W)
end

function variational_inference(X::Array{Float64}, prior::GaussianMixtureModel,
                               max_iter::Int)

    # Initialization
    Z_expt = init_latent_variable(size(X, 1), prior.K)
    posterior = update_posterior(Z_expt, X, prior)

    for iter in 1:max_iter
        # E-step
        Z_expt = update_z(X, posterior)

        # M-step
        posterior = update_posterior(Z_expt, X, prior)
    end

    return posterior, Z_expt
end

end

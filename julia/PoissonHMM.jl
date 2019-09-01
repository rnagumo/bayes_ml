
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

export PoissonHMMModel, sample_model, sample_data, VI

struct PoissonHMMModel
    K::Int
    a::Array{Float64, 1}  # K
    b::Array{Float64, 1}  # K
    alpha::Array{Float64, 1}  # K
    beta::Array{Float64, 2}  # K*K
end

struct SampledPoissonHMMModel
    K::Int
    lambda::Array{Float64, 1}  # K
    phi::Array{Float64, 1}  # K
    A::Array{Float64, 2}  # K*K
end

# -----------------------------------------------------------
# Utils
# -----------------------------------------------------------

eye(D::Int) = Matrix{Float64}(I, D, D)

function sqsum(mat::Array{Float64}, idx::Int)
    return dropdims(sum(mat, dims=idx), dims=idx)
end

function categorical_sample(p::Vector{Float64})
    """
    Convert sampled index to one-hot vector
    """
    s = zeros(length(p))
    s[rand(Categorical(p))] = 1
    return s
end

# -----------------------------------------------------------
# Sampling Functions
# -----------------------------------------------------------

function sample_model(model::PoissonHMMModel)
    """
    Sample paramters
    """
    # Dimension
    K = model.K

    # Sample lambda
    lambda = zeros(K)
    for k in 1:K
        lambda[k] = rand(Gamma(model.a[k], 1.0 / model.b[k]))
    end

    # Sample phi
    phi = rand(Dirichlet(model.alpha))

    # Sample A
    A = zeros(K, K)
    for k in 1:K
        A[:, k] = rand(Dirichlet(model.beta[:, k]))
    end

    return SampledPoissonHMMModel(K, lambda, phi, A)
end

function sample_data(N::Int, model::SampledPoissonHMMModel)
    """
    Sample data
    """
    # Dimension
    K = model.K

    # Sample S (latent variable)
    S = zeros(N, K)
    S[1, :] = categorical_sample(model.phi)
    for n in 2:N
        S[n, :] = categorical_sample(model.A[:, argmax(S[n - 1, :])])
    end

    # Sample X (data)
    X = zeros(N)
    for n in 1:N
        X[n] = rand(Poisson(model.lambda[argmax(S[n, :])]))
    end

    return X, S
end

# -----------------------------------------------------------
# Inference Functions
# -----------------------------------------------------------

function init_latent_variable(X::Array{Float64}, prior::PoissonHMMModel)
    # Dimension
    K = prior.K
    N = size(X, 1)

    S = reshape(rand(Dirichlet(ones(K) / K), N), (N, K))
    return S
end

function update_S(S::Array{Float64}, X::Array{Float64}, 
                  posterior::PoissonHMMModel)
    """
    Completely Factorized Variational Inference
    """
    # Dimension
    K = posterior.K
    N = size(X, 1)

    # Expectations
    lambda_expt = posterior.a ./ posterior.b
    ln_lambda_expt = digamma.(posterior.a) - log.(posterior.b)

    ln_lkh_expt = zeros(N, K)
    for n in 1:N
        ln_lkh_expt[n, :] = X[n] .* ln_lambda_expt - lambda_expt
    end

    ln_phi_expt = digamma.(posterior.alpha) .- digamma(sum(posterior.alpha))
    ln_A_expt = (digamma.(posterior.beta) 
                     .- digamma.(sum(posterior.beta, dims=1)))

    ln_S_expt = log.(S)

    # Update S (n=1~N)

    # n = 1
    ln_S_expt[1, :] = (ln_lkh_expt[1, :] 
                           + ln_A_expt * exp.(ln_S_expt[2, :])
                           + ln_phi_expt)
    ln_S_expt[1, :] .-= logsumexp(ln_S_expt[1, :])

    # n = 2 ~ N-1
    for n in 2:N - 1
        ln_S_expt[n, :] = (ln_lkh_expt[n, :]
                               + ln_A_expt * exp.(ln_S_expt[n - 1, :])
                               + ln_A_expt * exp.(ln_S_expt[n + 1, :]))
        ln_S_expt[n, :] .-= logsumexp(ln_S_expt[n, :])
    end

    # n = N
    ln_S_expt[N, :] = (ln_lkh_expt[N, :]
                               + ln_A_expt * exp.(ln_S_expt[N - 1, :]))
    ln_S_expt[N, :] .-= logsumexp(ln_S_expt[N, :])

    return exp.(ln_S_expt)
end

function update_lambda(S::Array{Float64}, X::Array{Float64},
                       prior::PoissonHMMModel, posterior::PoissonHMMModel)

    a = S' * X + prior.a
    b = sqsum(S, 1) + prior.b
    posterior = PoissonHMMModel(posterior.K, a, b, posterior.alpha,
                                posterior.beta)
    return posterior
end

function update_phi(S::Array{Float64}, prior::PoissonHMMModel, 
                    posterior::PoissonHMMModel)

    alpha = S[1, :] + prior.alpha
    posterior = PoissonHMMModel(posterior.K, posterior.a, posterior.b, alpha,
                                posterior.beta)
    return posterior
end

function update_A(S::Array{Float64}, prior::PoissonHMMModel, 
                  posterior::PoissonHMMModel)

    K = posterior.K
    N = size(S, 1)
    SS = zeros(K, K)
    for n in 2:N
        SS += S[n, :] * S[n - 1, :]'
    end

    beta = SS + prior.beta
    posterior = PoissonHMMModel(K, posterior.a, posterior.b, posterior.alpha,
                                beta)
    return posterior
end

function VI(X::Array{Float64}, prior::PoissonHMMModel, max_iter::Int)
    """
    Variational Inference for NMF
    """
    # Initialize latent variable
    S = init_latent_variable(X, prior)
    posterior = deepcopy(prior)

    # Inference
    for iter in 1:max_iter
        # E-step
        S = update_S(S, X, posterior)

        # M-step
        posterior = update_lambda(S, X, prior, posterior)
        posterior = update_phi(S, prior, posterior)
        posterior = update_A(S, prior, posterior)
    end

    return posterior, S
end

end

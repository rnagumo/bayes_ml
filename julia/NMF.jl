
module NMF

"""
Variational Inference for Bayesian NMF

p(X, S, W, H) = p(X|S) p(S|W, H) p(W) p(H)

p(X|S) = Del(X_d,n | sum_m(S_d,m,n))
p(S|W, H) = Poi(S_d,m,n | W_d,m, H_m,n)
p(W) = Gam(W_d,m | a_W, b_W)
p(H) = Gam(H_m,n | a_H, b_H)
"""

using LinearAlgebra
using Distributions
using StatsFuns
using SpecialFunctions

export NMFModel, VI, sample_data

struct NMFModel
    D::Int
    M::Int
    N::Int
    a_w::Array{Float64, 2}  # D*M
    b_w::Array{Float64, 1}  # M
    a_h::Array{Float64, 2}  # M*N
    b_h::Array{Float64, 1}  # M
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

function sample_data(N::Int, model::NMFModel)
    """
    Sample parameters and data
    """

    # Dimension
    D = model.D
    M = model.M
    N = model.N

    # Sample W
    W = zeros(D, M)
    for d in 1:D
        for m in 1:M
            W[d, m] = rand(Gamma(model.a_w[d, m], 1.0 / model.b_w[m]))
        end
    end

    # Sample H
    H = zeros(M, N)
    for m in 1:M
        for n in 1:N
            H[m, n] = rand(Gamma(model.a_h[m, n], 1.0 / model.b_h[m]))
        end
    end

    # Sample S
    S = zeros(D, M, N)
    for d in 1:D
        for m in 1:M
            for n in 1:N
                S[d, m, n] = W[d, m] * H[m, n]
            end
        end
    end

    # Sample X
    X = sqsum(S, 2)

    return X, S, W, H
end

function init_latent_variable(X::Array{Float64, 2}, prior::NMFModel)
    # Dimension
    D = prior.D
    M = prior.M
    N = prior.N

    S = zeros(D, M, N)
    for d in 1:D
        for m in 1:M
            for n in 1:N
                S[d, m, n] = (X[d, n] * prior.a_w[d, m] / prior.b_w[m]
                              * prior.a_h[m, n] / prior.b_h[m])
            end
        end
    end

    return S
end

function update_S(X::Array{Float64, 2}, posterior::NMFModel)
    # Dimension
    D = posterior.D
    M = posterior.M
    N = posterior.N

    # Latent variable
    S = zeros(D, M, N)
    for d in 1:D
        for n in 1:N
            ln_p = (digamma.(posterior.a_w[d, :]) - log.(posterior.b_w) 
                    + digamma.(posterior.a_h[:, n]) - log.(posterior.b_h))
            ln_p .-= logsumexp(ln_p)
            S[d, :, n] = X[d, n] .+ exp.(ln_p)
        end
    end

    return S
end

function update_W(S::Array{Float64, 3}, prior::NMFModel, posterior::NMFModel)
    # Dimension
    D = prior.D
    M = prior.M
    N = prior.N

    # Update
    a_w = prior.a_w + sqsum(S, 3)
    b_w = prior.b_w + sqsum(posterior.a_h, 2) .* posterior.b_h

    return NMFModel(D, M, N, a_w, b_w, posterior.a_h, posterior.b_h)
end

function update_H(S::Array{Float64, 3}, prior::NMFModel, posterior::NMFModel)
    # Dimension
    D = prior.D
    M = prior.M
    N = prior.N

    # Update
    a_h = prior.a_h + sqsum(S, 1)
    b_h = prior.b_h + sqsum(posterior.a_w, 1) .* posterior.b_w

    return NMFModel(D, M, N, posterior.a_w, posterior.b_w, a_h, b_h)
end

function VI(X::Array{Float64, 2}, prior::NMFModel, max_iter::Int)
    """
    Variational Inference for NMF
    """
    # Initialize latent variable
    S = init_latent_variable(X, prior)
    posterior = deepcopy(prior)

    # Inference
    for iter in 1:max_iter
        # Update S
        S = update_S(X, posterior)

        # Update parameters
        posterior = update_W(S, prior, posterior)
        posterior = update_H(S, prior, posterior)
    end

    return posterior, S
end

end

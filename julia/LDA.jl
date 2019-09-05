
module LDA

"""
Latent Dirichlet Allocation

p(W, Z, Phi, Theta) = p(W|Z, Phi) p(Z|Phi) p(Phi) p(Theta)

p(w_d,n| z_d,n, Phi) = Prod_{k=1}^{K} Cat(w_d,n|phi_k)^{z_d,n,k}
p(z_d,n| theta_d) = Cat(z_d,n| theta_d)
p(theta_d) = Dir(theta_d | alpha)
p(phi_k) = Dir(phi_k| beta)
"""

using LinearAlgebra
using Distributions
using StatsFuns
using SpecialFunctions

export LDAModel, sample_model, sample_data, VI

struct LDAModel
    D::Int  # Number of documents
    K::Int  # Number of topics
    V::Int  # Number of vocabulary
    alpha::Array{Float64, 2}  # D*K
    beta::Array{Float64, 2}  # K*V
end

struct SampledLDAModel
    D::Int
    K::Int
    V::Int
    theta::Array{Float64, 2}  # D*K
    phi::Array{Float64, 2}  # K*V
end

# -----------------------------------------------------------
# Utils
# -----------------------------------------------------------

eye(D::Int) = Matrix{Float64}(I, D, D)

function sqsum(mat, idx::Int)
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

function reverse_one_hot_vectorization(A::Array)

    X, Y, _ = size(A)
    B = zeros(Int32, X, Y)
    for x in 1:X
        for y in 1:Y
            B[x, y] = argmax(A[x, y, :])
        end
    end

    return B
end

# -----------------------------------------------------------
# Sampling Functions
# -----------------------------------------------------------

function sample_model(model::LDAModel)
    """
    Sample paramters
    """
    # Dimension
    D = model.D
    K = model.K
    V = model.V

    # Sample parameters
    theta = zeros(D, K)
    for d in 1:D
        theta[d, :] = rand(Dirichlet(model.alpha[d, :]))
    end

    phi = zeros(K, V)
    for k in 1:K
        phi[k, :] = rand(Dirichlet(model.beta[k, :]))
    end

    return SampledLDAModel(D, K, V, theta, phi)
end

function sample_data(N::Int, model::SampledLDAModel)
    """
    Sample data
    """
    # Dimension
    D = model.D

    # Sample Z (latent variable)
    Z = zeros(Int32, D, N)
    for d in 1:D
        for n in 1:N
            Z[d, n] = rand(Categorical(model.theta[d, :]))
        end
    end

    # Sample X (data)
    X = zeros(Int32, D, N)
    for d in 1:D
        for n in 1:N
            X[d, n] = rand(Categorical(model.phi[Z[d, n], :]))
        end
    end

    return X, Z
end

# -----------------------------------------------------------
# Variational Inference Functions
# -----------------------------------------------------------

function init_latent_variable(X::Array{Int32, 2}, prior::LDAModel)

    # Dimension
    N = size(X, 2)
    D = prior.D
    K = prior.K

    # Sample Z (latent variable)
    Z = zeros(D, N, K)
    for d in 1:D
        Z[d, :, :] = categorical_sample(ones(K) ./ K, N)
    end

    return Z
end

function update_latent_variable(X::Array{Int32, 2}, posterior::LDAModel)

    # Dimension
    N = size(X, 2)
    D = posterior.D
    K = posterior.K

    # Expectations
    ln_theta_expt = (digamma.(posterior.alpha) 
                         .- digamma.(sqsum(posterior.alpha, 2)))
    ln_phi_expt = (digamma.(posterior.beta) 
                       .- digamma.(sqsum(posterior.beta, 2)))

    # Update Z
    ln_Z_expt = zeros(D, N, K)
    for d in 1:D
        for n in 1:N
            ln_Z_expt[d, n, :] = (ln_phi_expt[:, X[d, n]]
                                      + ln_theta_expt[d, :])
            ln_Z_expt[d, n, :] .-= logsumexp(ln_Z_expt[d, n, :])
        end
    end

    return exp.(ln_Z_expt)
end

function update_posterior(Z::Array{Float64, 3}, X::Array{Int32, 2},
                          prior::LDAModel)

    alpha = sqsum(Z, 2) + prior.alpha

    beta = deepcopy(prior.beta)
    D, N, _ = size(Z)
    for d in 1:D
        for n in 1:N
            beta[:, X[d, n]] += Z[d, n, :]
        end
    end

    return LDAModel(prior.D, prior.K, prior.V, alpha, beta)
end

function VI(X::Array{Int32, 2}, prior::LDAModel, max_iter::Int)
    """
    Variational Inference for LDAModel
    """
    # Initialization
    Z_expt = init_latent_variable(X, prior)
    posterior = update_posterior(Z_expt, X, prior)

    # Inference
    for iter in 1:max_iter
        # E-step
        Z_expt = update_latent_variable(X, posterior)

        # M-step
        posterior = update_posterior(Z_expt, X, prior)
    end

    Z_expt = reverse_one_hot_vectorization(Z_expt)
    return posterior, Z_expt
end

end

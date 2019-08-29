
module DimensionalityReduction

"""
Variational Inference for DimensionalityReduction

p(Y, X, W, mu) = p(Y|X, W, mu)p(X)p(W)p(mu)

p(y_n|x_n, W, mu) = N(y_n|W'x_n + mu, sigma_y*I)
p(x_n) = N(x_n|0, I)
p(w_d) = N(w_d|0, sigma_w)
p(mu) = N(mu|0, sigma_mu)
"""

using LinearAlgebra
using Distributions

export DRModel

struct DRModel
    D::Int
    M::Int
    sigma2_y::Float64
    m_W::Array{Float64, 2}
    Sigma_W::Array{Float64, 3}
    m_mu::Array{Float64, 1}
    Sigma_mu::Array{Float64, 2}
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

function sample_data(N::Int, model::DRModel)
    """
    Sample paramters and data from model.
    """

    D = model.D
    M = model.M

    # Sample W (R^M*D)
    W = zeros(M, D)
    for d in 1:D
        W[:, d] = rand(MvNormal(model.m_W[:, d], model.Sigma_W[:, :, d]))
    end

    # Sample mu (R^D)
    mu = rand(MvNormal(model.m_mu, model.Sigma_mu))

    # Sample x (R^N*M)
    X = randn(N, M)

    # Sample y (R^N*D)
    Y = zeros(N, D)
    for n in 1:N
        Y[n, :] = rand(MvNormal(W'*X[n, :] + mu,
                       model.sigma2_y * eye(D)))
    end

    return Y, X, W, mu
end

function init(Y::Array{Float64, 2}, prior::DRModel)
    """
    Initalize latent variable
    """
    
    N, D = size(Y)
    M = prior.M
    
    X = randn(N, M)
    XX = zeros(N, M, M)
    for n in 1:N
        XX[n, :, :] = X[n, :] * X[n, :]' + eye(M)
    end

    return X, XX
end

function update_mu(Y::Array{Float64, 2}, X::Array{Float64, 2},
                   XX::Array{Float64, 3}, prior::DRModel, posterior::DRModel)
    N, D = size(Y)
    M = prior.M

    # Expectation
    W = posterior.m_W

    # Update
    Sigma_mu = inv(N * inv(prior.sigma2_y) * eye(D)
                   + inv(prior.Sigma_mu))
    m_mu = Sigma_mu * (inv(prior.sigma2_y) * sqsum(Y - X * W, 1)
                       + inv(prior.Sigma_mu) * prior.m_mu)

    return DRModel(D, M, prior.sigma2_y, posterior.m_W, posterior.Sigma_W,
                   m_mu, Sigma_mu)
end

function update_W(Y::Array{Float64, 2}, X::Array{Float64, 2},
                  XX::Array{Float64, 3}, prior::DRModel, posterior::DRModel)
    
    N, D = size(Y)
    M = prior.M

    # Expectation
    mu = posterior.m_mu

    # Update
    m_W = zeros(M, D)
    Sigma_W = zeros(M, M, D)

    for d in 1:D
        Sigma_W[:, :, d] = inv(inv(prior.sigma2_y) * sqsum(XX, 1)
                               + inv(prior.Sigma_W[:, :, d]))
        m_W[:, d] = (Sigma_W[:, :, d] 
                     * (inv(prior.sigma2_y) 
                        * X' * (Y[:, d] - mu[d] * ones(N, 1))
                        + inv(prior.Sigma_W[:, :, d]) * prior.m_W[:, d]))
    end

    return DRModel(D, M, prior.sigma2_y, m_W, Sigma_W, 
                   posterior.m_mu, posterior.Sigma_mu)
end

function update_X(Y::Array{Float64, 2}, posterior::DRModel)
    N, D = size(Y)
    M = posterior.M

    # Expectation
    mu = posterior.m_mu
    W = posterior.m_W
    WW = zeros(M, M, D)
    for d in 1:D
        WW[:, :, d] = W[:, d] * W[:, d]' + posterior.Sigma_W[:, :, d]
    end

    # Update
    X = zeros(N, M)
    XX = zeros(N, M, M)
    Sigma = inv(inv(posterior.sigma2_y) * sqsum(WW, 3) + eye(M))
    for n in 1:N
        X[n, :] = inv(posterior.sigma2_y) * Sigma * W * (Y[n, :] - mu)
        XX[n, :, :] = X[n, :] * X[n, :]' + Sigma
    end

    return X, XX
end

function interpolate(mask::BitArray{2}, X::Array{Float64, 2},
                     posterior::DRModel)

    Y_est = X * posterior.m_W + repeat(posterior.m_mu', size(X, 1), 1)
    return Y_est[mask]
end

function VI(Y::Array{Float64, 2}, prior::DRModel, max_iter::Int)
    # Initalize latent variable
    X, XX = init(Y, prior)

    # Check missing values
    mask = isnan.(Y)

    posterior = deepcopy(prior)

    # Inference
    for iter in 1:max_iter
        if any(mask)
            Y[mask] = interpolate(mask, X, posterior)
        end

        # M-step
        posterior = update_W(Y, X, XX, prior, posterior)
        posterior = update_mu(Y, X, XX, prior, posterior)

        # E-step
        X, XX = update_X(Y, posterior)
    end

    return posterior, X
end

end

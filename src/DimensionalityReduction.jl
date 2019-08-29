
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

function sample_data(N::Int, model::DRModel)
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
                    model.sigma2_y * Matrix{Float64}(I, D, D)))
    end

    return Y, X, W, mu
end


end

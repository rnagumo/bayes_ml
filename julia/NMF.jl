
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

export NMFModel

struct NMFModel
    D::Int
    M::Int
    N::Int
    a_w::Array{Float64, 2}  # D*M
    a_h::Array{Float64, 2}  # M*N
    b_w::Array{Float64, 1}  # M
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

end

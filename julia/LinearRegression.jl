
module LinearRegression

"""
Linear Regression with unknown weight and uncertainty

p(Y, X, W, sg2) = p(Y|X, W, sg2) p(W, sg2)

p(Y|X, w, sg2) = N(Y|X * w, sg2 * I)
p(w, sg2) = NIG(w, sg2|m, V, a, b) = N(w|m, sg2 * V) IG(sg2|a, b)
"""

using LinearAlgebra
using Distributions
using StatsFuns
using SpecialFunctions

struct LinearRegressionModel
    """
    Hyper-parameters

    Dimensions
    Input: D
    Output: 1
    """
    D::Int
    m::Array{Float64, 1}
    V::Array{Float64, 2}
    a::Float64
    b::Float64
end

struct SampledLinearRegressionModel
    D::Int
    w::Array{Float64, 1}  # D
    sg2::Float64
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

function basis_func(x::Array{Float64, 1}, D::Int)
    X = repeat(x, 1, D) .^ (0:D - 1)'
    return X
end

# -----------------------------------------------------------
# Sampling Functions
# -----------------------------------------------------------

function sample_model(model::LinearRegressionModel)
    """
    Sample paramters
    """
    # Dimension
    D = model.D

    # Sample parameters
    sg2 = rand(InverseGamma(model.a, model.b))
    w = rand(MvNormal(model.m, sg2 .* model.V))

    return SampledLinearRegressionModel(D, w, sg2)
end

function sample_data(x::Array{Float64, 1},
                     model::SampledLinearRegressionModel)
    """
    Sample data
    """
    # Dimension
    N = size(x, 1)

    # Convert x
    X = basis_func(x, model.D)

    # Sample Y
    Y = rand(MvNormal(X * model.w, model.sg2 .* eye(N)))

    return Y
end

function sample_prediction(x::Array{Float64, 1},
                           model::LinearRegressionModel)
    """
    Sample prediction Y given X

    FIXME: Sample from MvTDist does not work after inference.
    message: matrix is not Hermitian; Cholesky factorization failed.
    """
    # Dimension
    N = size(x, 1)
    D = model.D
    X = basis_func(x, D)

    # Calculate parameters
    nu = 2 * model.a
    mu = X * model.m
    Lambda = (model.b / model.a) .* (eye(N) + X * model.V * X')

    # Sample Y
    Y = rand(MvTDist(nu, mu, Lambda))

    return Y, SampledTDistModel(D, nu, mu, Lambda)
end

# -----------------------------------------------------------
# Inference Functions
# -----------------------------------------------------------

function inference(x::Array{Float64, 1}, y::Array{Float64, 1},
                   prior::LinearRegressionModel)
    """
    Inference posterior
    """
    # Initialization
    N = size(x, 1)
    D = prior.D
    X = basis_func(x, D)

    # Inference
    V = inv(X' * X + inv(prior.V))
    m = V * (X' * y + inv(prior.V) * prior.m)
    a = prior.a + N / 2
    b = prior.b + 0.5 * (y' * y + prior.m' * inv(prior.V) * prior.m
                             - m' * inv(V) * m)

    return LinearRegressionModel(D, m, V, a, b)
end

end

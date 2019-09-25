
module LogisticRegression

"""
Logistic Regression with gradient descent

p(Y, X, W) = p(Y|X, W) p(W)

p(Y|X, W) = prod_{n=1}^{N]
    p(y_n|x_n, W) = Cat(y_n|f(W, x_n))
p(W) = prod_{m=1}^{M} prod_{d=1}^{D} N(w_{m,d}|0, lambda^{-1})
f(W, x_n) = SoftMax(W'*x_n)
"""

using LinearAlgebra
using Distributions
using StatsFuns
using SpecialFunctions

export LogisticRegressionModel, sample_model, sample_data, VI

struct LogisticRegressionModel
    M::Int
    lambda::Int
end

struct SampledLogisticRegressionModel
    M::Int
    W::Array{Float64, 1}  # M
end

struct ApproximatedLogisticRegressionModel
    """
    Gaussian approximation
    """
    M::Int
    mu::Array{Float64, 1}  # M
    sigma::Array{Float64, 1}  # M
end

# -----------------------------------------------------------
# Utils
# -----------------------------------------------------------

eye(D::Int) = Matrix{Float64}(I, D, D)

function sqsum(mat::Array{Float64}, idx::Int)
    return dropdims(sum(mat, dims=idx), dims=idx)
end

function sigmoid(x)
    return 1.0 ./ (1.0 .+ exp.(x))
end

function rho2sigma(x)
    return log.(1.0 .+ exp.(x))
end

# -----------------------------------------------------------
# Sampling Functions
# -----------------------------------------------------------

function sample_model(model::LogisticRegressionModel)
    """
    Sample paramters
    """
    # Dimension
    M = model.M

    # Sample w
    W = rand(MvNormal(zeros(M), model.lambda .* eye(M)))

    return SampledLogisticRegressionModel(M, W)
end

function sample_data(X::Array{Float64, 2},
                     model::SampledLogisticRegressionModel)
    """
    Sample data
    """
    # Dimension
    N = size(X, 1)
    M = model.M

    # Sample Y
    Y = zeros(N)
    for n in 1:N
        Y[n] = rand(Bernoulli(sigmoid(model.W' * X[n, :])))
    end

    return Y
end

function sample_prediction(X::Array{Float64, 2},
                           model::ApproximatedLogisticRegressionModel,
                           L::Int=10)
    """
    Sample prediction Y given X and approximated posterior

    L: Number of Monte Carlo sample "w"
    """
    # Dimension
    N, M = size(X)

    # Sample w and Y
    W = zeros(N, M)
    Y = zeros(N)
    for n in 1:N
        for l in 1:L
            w_tmp = rand(MvNormal(model.mu, diagm(0 => model.sigma .^ 2)))
            W[n, :] += w_tmp ./ L
            Y[n] += sigmoid(w_tmp' * X[n, :])[1] / L
        end
    end

    return Y, W
end

# -----------------------------------------------------------
# Inference Functions
# -----------------------------------------------------------

function VI(X::Array{Float64}, Y::Array{Float64},
            prior::LogisticRegressionModel, max_iter::Int=5, lr::Float64=0.001)
    """
    Gradient descent
    """
    # Initialization
    M = prior.M
    mu = randn(M)
    rho = randn(M)

    # Inference
    for iter in 1:max_iter
        # Sample epsilon
        ep = randn(M)
        W_tmp = mu + rho2sigma.(rho) .* ep

        # Calculate gradient
        d_mu = prior.lambda .* W_tmp + X' * (sigmoid.(X * W_tmp) - Y)
        d_rho = (-1 ./ rho2sigma.(rho) + prior.lambda .* W_tmp .* ep
                     + X' * (sigmoid.(X * W_tmp) - Y) .* ep) .* sigmoid.(rho)

        # Update variational paramters
        mu .-= lr * d_mu
        rho .-= lr * d_rho
    end

    return ApproximatedLogisticRegressionModel(M, mu, rho2sigma.(rho))
end

end

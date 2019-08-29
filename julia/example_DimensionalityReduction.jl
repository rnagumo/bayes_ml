
"""
Example for DimensionalityReduction
"""

using LinearAlgebra
using PyPlot, PyCall
using ArgParse
using Distributions

push!(LOAD_PATH, ".")
using DimensionalityReduction

# -----------------------------------------------------------
# Executing function
# -----------------------------------------------------------

function check_with_dummy_data()

    # Dimension
    D = 20
    M = 16

    # Hyper-parameters
    sigma2_y = 0.001
    m_w = zeros(M, D)
    Sigma_w = zeros(M, M, D)
    for d in 1:D
        Sigma_w[:, :, d] = 0.1 * Matrix{Float64}(I, M, M)
    end
    m_mu = zeros(D)
    Sigma_mu = Matrix{Float64}(I, D, D)
    
    # Prior
    prior = DimensionalityReduction.DRModel(D, M, sigma2_y, m_w, Sigma_w,
                                            m_mu, Sigma_mu)

    # Sample dummy data
    Y, X, W, mu = DimensionalityReduction.sample_data(20, prior)

    # Inference
    max_iter = 2
    posterior, X = DimensionalityReduction.VI(Y, prior, max_iter)
end

function test_iris()
    # Load data
    datasets = pyimport("sklearn.datasets")
    X, y = datasets.load_iris(return_X_y=true)
    N, D = size(X)

    # Hyper-parameters
    M = 2
    sigma2_y = 0.001
    m_w = zeros(M, D)
    Sigma_w = zeros(M, M, D)
    for d in 1:D
        Sigma_w[:, :, d] = 0.1 * Matrix{Float64}(I, M, M)
    end
    m_mu = zeros(D)
    Sigma_mu = Matrix{Float64}(I, D, D)
    
    # Prior
    prior = DimensionalityReduction.DRModel(D, M, sigma2_y, m_w, Sigma_w,
                                            m_mu, Sigma_mu)

    # Inference
    max_iter = 100
    posterior, Z_est = DimensionalityReduction.VI(deepcopy(X), prior, max_iter)

    # Visualize
    figure("Iris data")
    clf()
    scatter(Z_est[y .== 0, 1], Z_est[y .== 0, 2], color="r")
    scatter(Z_est[y .== 1, 1], Z_est[y .== 1, 2], color="g")
    scatter(Z_est[y .== 2, 1], Z_est[y .== 2, 2], color="b")
    try
        show()
    catch
        close()
    end
end

function test_face_missing()
    # Load data
    datasets = pyimport("sklearn.datasets")
    dataset = datasets.fetch_olivetti_faces(shuffle=true)
    Y = deepcopy(dataset["data"])
    N, D = size(Y)

    # Mask
    missing_rate = 0.50
    mask = rand(Uniform(), size(Y)) .< missing_rate
    Y_obs = deepcopy(Y)
    Y_obs[mask] .= NaN
    Y_obs = convert(Array{Float64, 2}, Y_obs)

    # Hyper-parameters
    M = 16
    sigma2_y = 0.001
    m_w = zeros(M, D)
    Sigma_w = zeros(M, M, D)
    for d in 1:D
        Sigma_w[:, :, d] = 0.1 * Matrix{Float64}(I, M, M)
    end
    m_mu = zeros(D)
    Sigma_mu = Matrix{Float64}(I, D, D)
    
    # Prior
    prior = DimensionalityReduction.DRModel(D, M, sigma2_y, m_w, Sigma_w,
                                            m_mu, Sigma_mu)

    # Inference
    max_iter = 100
    posterior, Z_est = DimensionalityReduction.VI(deepcopy(Y_obs), prior,
                                                  max_iter)
    Y_est = Z_est * posterior.m_W + repeat(posterior.m_mu', size(Z_est, 1), 1)
    Y_itp = deepcopy(Y_obs)
    Y_itp[mask] = Y_est[mask]

    # Visualize
    figure("Olivetti faces")
    clf()
    for i in 1:12
        subplot(4, 6, i)
        imshow(dataset["images"][i, :, :])
        xticks([])
        yticks([])
    end

    for i in 1:12
        subplot(6, 6, i + 12)
        imshow(reshape(Y_obs[i, :], 64, 64)')
        xticks([])
        yticks([])
    end

    for i in 1:12
        subplot(6, 6, i + 24)
        imshow(reshape(Y_itp[i, :], 64, 64)')
        xticks([])
        yticks([])
    end

    try
        show()
    catch
        close()
    end

end

# -----------------------------------------------------------
# Main function
# -----------------------------------------------------------

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "func"
            help = "selected function"
            arg_type = String
            default = "dummy"
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    # Function hash table
    func_dict = Dict([("dummy", check_with_dummy_data),
                      ("iris", test_iris),
                      ("face", test_face_missing)])

    # Execute selected function
    func_dict[parsed_args["func"]]()
end

main()


using LinearAlgebra
using PyPlot, PyCall
using ArgParse

# @pyimport sklearn.datasets as datasets

push!(LOAD_PATH, ".")
using DimensionalityReduction

# -----------------------------------------------------------
# Executing function
# -----------------------------------------------------------

function visualize()
    x = range(0; stop=2pi, length=1000)
    y = sin.(3 * x + 4 * cos.(2 * x))
    figure()
    plot(x, y)
    show()
end

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

    # Execute selected function
    if parsed_args["func"] == "visualize"
        visualize()
    elseif parsed_args["func"] == "dummy"
        check_with_dummy_data()
    end
end

main()

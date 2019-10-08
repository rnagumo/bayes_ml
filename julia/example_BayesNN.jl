
"""
Example for BayesNeuralNet
"""

using LinearAlgebra
using PyPlot, PyCall
using ArgParse
using Distributions

push!(LOAD_PATH, ".")
import BayesNeuralNet

# -----------------------------------------------------------
# Executing function
# -----------------------------------------------------------

function check_with_dummy_data()

    #Prior
    M = 2
    K = 2
    D = 1
    sigma2_w = 1.0
    sigma2_y = 1.0
    prior = BayesNeuralNet.BayesNNModel(M, K, D, sigma2_w, sigma2_y)

    # Sample data
    x = collect(range(0.0, stop=10.0, length=20))
    Y, Y_obs, nn_model = BayesNeuralNet.sample_data_from_prior(prior, x)

    println(size(Y))
    println(size(Y_obs))
end

function main_prior_plot()

    # -------------------------------------
    # Data
    # -------------------------------------

    #Prior
    M = 2
    K = 2
    D = 1
    sigma2_w = 100.0
    sigma2_y = 0.1
    prior = BayesNeuralNet.BayesNNModel(M, K, D, sigma2_w, sigma2_y)

    # -------------------------------------
    # Visualization
    # -------------------------------------

    figure(figsize=(8, 6))

    x = collect(range(-5.0, stop=5.0, length=1000))
    for n in 1:10
        Y, _, _ = BayesNeuralNet.sample_data_from_prior(prior, x)
        plot(x, Y)
    end

    xlabel("x")
    ylabel("y")
    title("K=$K, sigma2_w=$sigma2_w")
    tight_layout()
    savefig("../results/nn_sample-K_$K-sigma2_w_$sigma2_w.png")

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
                      ("prior", main_prior_plot)])

    # Execute selected function
    func_dict[parsed_args["func"]]()
end

main()

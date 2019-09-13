
"""
Example for Bayesian Linear Regression
"""

using LinearAlgebra
using PyPlot, PyCall
using ArgParse
using Distributions
using DataFrames

push!(LOAD_PATH, ".")
import LinearRegression

# -----------------------------------------------------------
# Executing function
# -----------------------------------------------------------

eye(D::Int) = Matrix{Float64}(I, D, D)

function check_with_dummy_data()

    # Prior
    D = 2
    m = rand(D)
    V = eye(D)
    a = 1.0
    b = 1.0
    prior = LinearRegression.LinearRegressionModel(D, m, V, a, b)

    # Sample dummy data
    sampled_model = LinearRegression.sample_model(prior)
    x = collect(range(0.0, stop=10.0, length=20))
    y = LinearRegression.sample_data(x, sampled_model)

    # Inference
    posterior = LinearRegression.inference(x, y, prior)

    # Prediction
    y_pred, sampled_posterior = LinearRegression.sample_prediction(
                                    x, posterior)

    println(y)
    println(y_pred)
end

function plot_2d_gauss()

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
                      ("plot", plot_2d_gauss)])

    # Execute selected function
    func_dict[parsed_args["func"]]()
end

main()


"""
Example for LogisticRegression
"""

using LinearAlgebra
using PyPlot, PyCall
using ArgParse
using Distributions
using JLD

push!(LOAD_PATH, ".")
import LogisticRegression

# -----------------------------------------------------------
# Executing function
# -----------------------------------------------------------

function check_with_dummy_data()

    # Prior
    M = 2
    lambda = 100.0    
    prior = LogisticRegression.LogisticRegressionModel(M, lambda)
    sampled_model = LogisticRegression.sample_model(prior)

    # Sample dummy data
    N = 5
    X = 2 .* rand(N, M) .- 1.0  # Input (N*M)
    Y = LogisticRegression.sample_data(X, sampled_model)  # Output (N*1)

    # Inference
    max_iter = 1
    posterior = LogisticRegression.VI(X, Y, prior)

    # Predict
    Y_pred, W_pred = LogisticRegression.sample_prediction(X, posterior)

    # println(X, Y)
    println(Y_pred)
    println(W_pred)
end

function predict_with_plot()
    # -------------------------------------------------------------
    # Data & modeling
    # -------------------------------------------------------------
    
    # Prior
    M = 2
    lambda = 100.0    
    prior = LogisticRegression.LogisticRegressionModel(M, lambda)
    sampled_model = LogisticRegression.sample_model(prior)

    # Sample dummy data
    N = 50
    X = 2 .* rand(N, M) .- 1.0  # Input (N*M)
    Y = LogisticRegression.sample_data(X, sampled_model)  # Output (N*1)

    # Inference
    max_iter = 1
    lr = 1e-4
    posterior = LogisticRegression.VI(X, Y, prior, max_iter, lr)

    # -------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------

    N = 100
    R = 100
    xmin = 2 * minimum(X[:, 1])
    xmax = 2 * maximum(X[:, 1])
    ymin = minimum(X[:, 2])
    ymax = maximum(X[:, 2])

    x1 = range(xmin, stop=xmax, length=R)
    x2 = range(ymin, stop=ymax, length= R)
    x1grid = repeat(x1, 1, R)
    x2grid = repeat(x2', R, 1)
    X_pred = [x1grid[:] x2grid[:]]

    # Predict
    Z_pred, W_pred = LogisticRegression.sample_prediction(X_pred, posterior)
    Z_grid = reshape(Z_pred, R, R)

    # Plot
    figure()
    contour(x1grid, x2grid, Z_grid, alpha=0.5, cmap=get_cmap("bwr"))
    scatter(X[Y .== 1, 1]', X[Y .== 1, 2]', c="r")
    scatter(X[Y .== 0, 1]', X[Y .== 0, 2]', c="b")

    for n in 1:minimum([10, N])
        y1 = - xmin * W_pred[n, 1] / W_pred[n, 2]
        y2 = - xmax * W_pred[n, 1] / W_pred[n, 2]
        # plot([xmin, xmax], [y1, y2], c="k")
    end

    xlim([xmin, xmax])
    ylim([ymin, ymax])

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
                      ("plot", predict_with_plot)])

    # Execute selected function
    func_dict[parsed_args["func"]]()
end

main()

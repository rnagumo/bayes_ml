
"""
Example for MCMC
"""

using LinearAlgebra
using PyPlot, PyCall
using ArgParse
using Distributions

push!(LOAD_PATH, ".")
import MCMC

# -----------------------------------------------------------
# Executing function
# -----------------------------------------------------------

function check_with_dummy_data()
    # Target Gaussian distribution
    mu = [0.0, 0.0]
    Sigma = [1 0.9; 0.9 1]
    max_iter = 5

    # Sample
    samples_mh, acpt_rate_mh = MCMC.metropolis_hastings(mu, Sigma, max_iter)
    samples_hmc, acpt_rate_hmc = MCMC.hamiltonian_montecarlo(
        mu, Sigma, max_iter)
    samples_gibbs = MCMC.gibbs_sampling(mu, Sigma, max_iter)

    println(size(samples_mh), acpt_rate_mh)
    println(size(samples_hmc), acpt_rate_hmc)
    println(size(samples_gibbs))
end

function test_2d_plot()
    # Target Gaussian distribution
    mu = [0.0, 0.0]
    Sigma = [1 0.9; 0.9 1]
    max_iter = 100

    samples_mh, acpt_rate_mh = MCMC.metropolis_hastings(mu, Sigma, max_iter)
    samples_hmc, acpt_rate_hmc = MCMC.hamiltonian_montecarlo(
        mu, Sigma, max_iter, 0.25, 25, 100)
    samples_gibbs = MCMC.gibbs_sampling(mu, Sigma, max_iter)

    # -------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------

    p_pdf = gauss_pdf_1s(mu, Sigma)

    # Plot
    figure(figsize=(12, 6))

    subplot(131)
    plot(p_pdf[:, 1], p_pdf[:, 2], alpha=0.9)
    plot(samples_mh[:, 1], samples_mh[:, 2], "--o", alpha=0.6,
         label="acceptance ratio = $acpt_rate_mh")
    legend(loc="upper right")
    title("Metropolis Hastings")

    subplot(132)
    plot(p_pdf[:, 1], p_pdf[:, 2], alpha=0.9)
    plot(samples_hmc[:, 1], samples_hmc[:, 2], "--o", alpha=0.6,
         label="acceptance ratio = $acpt_rate_hmc")
    legend(loc="upper right")
    title("Hamiltonian Monte Carlo")

    subplot(133)
    plot(p_pdf[:, 1], p_pdf[:, 2], alpha=0.9)
    plot(samples_gibbs[:, 1], samples_gibbs[:, 2], "--o", alpha=0.6,
         label="acceptance ratio = 1.0")
    legend(loc="upper right")
    title("Gibbs sampling")

    tight_layout()
    savefig("../results/mcmc.png")

    try
        show()
    catch
        close()
    end
end

function gauss_pdf_1s(mu::Array, Sigma::Array, res::Int=100, c::Float64=1.0)
    """
    Return Gaussian pdf (probability density function) in 1 sigma

    Parameters
    ----------
    mu : Array
        Mean of Gaussian

    Sigma : Array
        Covariance of Gaussian

    res : Int
        Resolution of curve

    c : Float
        Scaling coefficient

    Returns
    -------
    P : Array, size(res + 1, 2)
    """
    # Eigenvalues, eigenvectors
    val, vec = eigen(Sigma)
    w = 2 * pi / res .* (0:res)

    a = sqrt(c * val[1])
    b = sqrt(c * val[2])
    p1 = a .* cos.(w)
    p2 = b .* sin.(w)
    P = repeat(mu', res + 1, 1) + vcat(p1', p2')' * vec

    return P
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
                      ("plot", test_2d_plot)])

    # Execute selected function
    func_dict[parsed_args["func"]]()
end

main()

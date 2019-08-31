
"""
Example for NMF (Non-negative Matrix Factorization)
"""

using LinearAlgebra
using PyPlot, PyCall
using ArgParse
using Distributions
using DataFrames

push!(LOAD_PATH, ".")
import NMF

# -----------------------------------------------------------
# Executing function
# -----------------------------------------------------------

function check_with_dummy_data()

    # Dimension
    D = 2
    M = 4
    N = 2

    # Hyper-parameters
    a_w = rand(D, M)
    b_w = rand(M)
    a_h = rand(M, N)
    b_h = rand(M)
    
    # Prior
    prior = NMF.NMFModel(D, M, N, a_w, b_w, a_h, b_h)

    # Sample dummy data
    X, S, W, H = NMF.sample_data(20, prior)

    # Inference
    max_iter = 1
    posterior, S_est = NMF.VI(X, prior, max_iter)

    println(S)
    println(S_est)
end

function audio_decomposition()
    # Load data
    wf = pyimport("scipy.io.wavfile")
    wavfile = "../data/organ.wav"
    fs, data = wf.read(wavfile)

    figure("data")
    clf()
    Pxx, freqs, t, im = specgram(data.T[0], Fs=fs, NFFT=256,
                                 noverlap=0)
    xlabel("time [sec]")
    ylabel("frequency [Hz]")
    ylim([0, 22000])
    savefig("../results/spec.png")
    close()

    # Model
    # Dimension
    D, N = size(Pxx)
    M = 2

    # Hyper-parameters
    a_w = ones(D, M)
    b_w = ones(M)
    a_h = ones(M, N)
    b_h = 100.0 * ones(M)
    
    # Prior
    prior = NMF.NMFModel(D, M, N, a_w, b_w, a_h, b_h)

    # Inference
    max_iter = 50
    posterior, S_est = NMF.VI(Float64.(round.(Pxx)), prior, max_iter)

    W_est = posterior.a_w ./ repeat(posterior.b_w', D, 1)
    H_est = posterior.a_h ./ repeat(posterior.b_h, 1, N)

    # Visualize
    figure("W")
    clf()
    for m in 1:M
        subplot(M, 1, m)
        plot(W_est[:, m])
        xlim([0, D])
        ylim([0, ylim()[2]])
    end
    savefig("../results/nmf_w.png")
    close()

    figure("H")
    clf()
    for m in 1:M
        subplot(M, 1, m)
        plot(H_est[m, :])
        xlim([0, N])
        ylim([0, ylim()[2]])
    end
    savefig("../results/nmf_h.png")
    close()

    println("Finish")
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
                      ("audio", audio_decomposition)])

    # Execute selected function
    func_dict[parsed_args["func"]]()
end

main()

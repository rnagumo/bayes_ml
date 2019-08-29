
using LinearAlgebra
using PyPlot

push!(LOAD_PATH, ".")
using DimensionalityReduction

function visualize()
    x = range(0; stop=2pi, length=1000)
    y = sin.(3 * x + 4 * cos.(2 * x))
    figure()
    plot(x, y)
    show()
end

function main()

    # Dimension
    D = 20
    M = 16

    # Parameters
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

    DimensionalityReduction.sample_data(20, prior)
end

# visualize()
main()

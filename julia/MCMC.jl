
module MCMC

"""
MCMC with 2d Gaussian
"""

using LinearAlgebra
using Distributions
using StatsFuns
using SpecialFunctions

eye(D::Int) = Matrix{Float64}(I, D, D)

function metropolis_hastings(mu::Array, Sigma::Array, max_iter::Int)
    """
    Metroplis Hastings with Gaussian kernel

    In this case, this is also called random-walk Metropolis
    """

    # Dimension
    D = size(mu, 1)

    # Sample
    sample_list = zeros(max_iter, D)
    z0 = randn(D)

    # Acceptance count
    acpt_cnt = 0

    for iter in 1:max_iter
        # Sample new state from proposal
        # Nomral Gauss (random-walk), symmetric q(z1|z0) = q(z0|z1)
        # In this case, MH equals to random-walk Metroplis
        z1 = rand(MvNormal(z0, eye(D)))

        # Calculate unnormalized probability of target distribution
        # with Gaussian kernel
        z0_dist = exp(-0.5 * (z0 - mu)' * Sigma * (z0 - mu))
        z1_dist = exp(-0.5 * (z1 - mu)' * Sigma * (z1 - mu))
        
        if rand() <= min(1, z1_dist / z0_dist)
            # Accept
            acpt_cnt += 1
            sample_list[iter, :] = z1
            z0 = z1
        else
            # Reject
            sample_list[iter, :] = z0
        end
    end

    return sample_list, round(acpt_cnt / max_iter, digits=2)
end

function hamiltonian_montecarlo(mu::Array, Sigma::Array, max_iter::Int,
                                epsilon::Float64=1.0, L::Int=2, burnin::Int=5)
    """
    Hamiltonian Monte Carlo

    ref) MCMC using Hamiltonian dynamics
    http://arxiv.org/abs/1206.1901
    """

    # Dimension
    D = size(mu, 1)

    # Potential energy
    # q is the target variable.
    # In practice, U is negative unnormalized log posterior with T=1
    # i.e., U(q) = -log P(q|D) = -log P(q) - log P(D|q) + const
    # although const is often neglected.
    # In this case, U(q) = -log N(q|mu, Sigma)
    U(q::Array) = (q - mu)' * inv(Sigma) * (q - mu) / 2
    grad_U(q::Array) = inv(Sigma) * (q - mu)

    # Kinetic energy (T = 1, M = I)
    # K(p) = -log N(p|0, 1)
    K(p::Array) = p' * p / 2
    grad_K(p::Array) = p
    
    # Sample
    sample_list = zeros(max_iter + burnin, D)
    current_q = randn(D)
    
    # Acceptance count
    acpt_cnt = 0

    for iter in 1:(max_iter + burnin)
        # Initialization
        current_p = randn(D)
        q = deepcopy(current_q)
        p = deepcopy(current_p)

        # HMC Step
        # 1. Half step for momentum
        p = p - epsilon .* grad_U(q) ./ 2

        # 2. L step for momentum and position
        for i in 1:L
            # Full step for position
            q = q + epsilon .* grad_K(p)

            # Full step for momentum, except last step
            if i != L
                p = p - epsilon .* grad_U(q)
            end
        end

        # 3. Final half step for momentum
        p = p - epsilon .* grad_U(q) ./ 2

        # Negate momentum at end of trajectory
        p = -p

        # Evaluate potential
        current_H = U(current_q) + K(current_p)
        proposed_H = U(q) + K(p)

        if rand() <= min(1, exp(-proposed_H + current_H))
            # Accept
            sample_list[iter, :] = q
            current_q = q
            if iter > burnin
                acpt_cnt += 1
            end
        else
            # Reject
            sample_list[iter, :] = current_q
        end
    end

    sample_list = sample_list[burnin:max_iter + burnin, :]

    return sample_list, round(acpt_cnt / max_iter, digits=2)
end

function gibbs_sampling(mu::Array, Sigma::Array, max_iter::Int)
    # Dimension
    D = size(mu, 1)

    # Sample
    sample_list = zeros(max_iter * 2, D)
    z0 = randn(D)

    current = deepcopy(z0)
    for iter in 1:max_iter
        # Sample z1 from conditional Gaussian
        mu_12 = mu[1] + Sigma[1, 2] * inv(Sigma[2, 2]) * (current[2] - mu[2])
        sigma_12 = Sigma[1, 1] - Sigma[1, 2] * inv(Sigma[2, 2]) * Sigma[2, 1]
        z1 = rand(Normal(mu_12, sigma_12))
        current[1] = z1
        sample_list[iter * 2 - 1, :] = current

        # Sample z2 from conditional Gaussian
        mu_21 = mu[2] + Sigma[2, 1] * inv(Sigma[1, 1]) * (current[1] - mu[1])
        sigma_21 = Sigma[2, 2] - Sigma[2, 1] * inv(Sigma[1, 1]) * Sigma[1, 2]
        z2 = rand(Normal(mu_21, sigma_21))
        current[2] = z2
        sample_list[iter * 2, :] = current
    end

    return sample_list
end

end

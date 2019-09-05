
"""
Example for LDA (Latent Dirichlet Allocation)
"""

using LinearAlgebra
using PyPlot, PyCall
using ArgParse
using Distributions

push!(LOAD_PATH, ".")
import LDA

# -----------------------------------------------------------
# Executing function
# -----------------------------------------------------------

function check_with_dummy_data()

    # Prior
    D = 2  # Number of documents
    K = 2  # Number of topics
    V = 10  # Number of vocabulary
    alpha = ones(D, K)
    beta = ones(K, V)

    prior = LDA.LDAModel(D, K, V, alpha, beta)

    # Sample dummy data
    sampled_model = LDA.sample_model(prior)
    X, S = LDA.sample_data(5, sampled_model)

    # Inference
    max_iter = 1
    posterior, S_est = LDA.VI(X, prior, max_iter)

    println(S)
    println(S_est)
end

function newsgroup_example()
    # Load dataset
    datasets = pyimport("sklearn.datasets")
    categories = ["alt.atheism", "soc.religion.christian", 
                  "comp.graphics", "sci.med"]
    twenty_train = datasets.fetch_20newsgroups(
        subset="train", categories=categories,
        shuffle=true, random_state=0)
    
    # Convert words to vector
    np = pyimport("numpy")
    fe_text = pyimport("sklearn.feature_extraction.text")
    cnt_vect = fe_text.CountVectorizer(dtype=np.int32)   
    X_train_counts = cnt_vect.fit_transform(twenty_train["data"])
    
    # Prior
    D = X_train_counts.shape[1]  # Number of documents
    K = 4  # Number of topics
    V = X_train_counts.shape[2]  # Number of vocabulary
    alpha = ones(D, K)
    beta = ones(K, V)
    prior = LDA.LDAModel(D, K, V, alpha, beta)

    # Inference
    max_iter = 1
    posterior, S_est = LDA.VI(X_train_counts.toarray()[1:20, :],
                              prior, max_iter)
    
    return twenty_train, posterior, S_est
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
                      ("news", newsgroup_example)])

    # Execute selected function
    func_dict[parsed_args["func"]]()
end

main()

"""
Price equation and Fisher's fundamental theorem verification utilities.
"""

function price_equation_check(risks::Vector{Float64}, fitnesses::Vector{Float64},
                               mean_fitness::Float64)
    # Price equation: Delta R_bar ~ -Cov(w, R) / w_bar
    # Using population (not sample) covariance: corrected=false
    K = length(risks)
    mean_r = mean(risks)
    mean_w = mean_fitness

    cov_wr = 0.0
    for i in 1:K
        cov_wr += (fitnesses[i] - mean_w) * (risks[i] - mean_r)
    end
    cov_wr /= K

    selection_term = -cov_wr / mean_w
    var_risk = 0.0
    for i in 1:K
        var_risk += (risks[i] - mean_r)^2
    end
    var_risk /= K

    return (selection_term=selection_term, cov_wr=cov_wr,
            mean_risk=mean_r, var_risk=var_risk)
end

function fishers_theorem_check(fitnesses::Vector{Float64}, mean_fitness::Float64)
    # Fisher's fundamental theorem: Delta w_bar ~ Var(w) / w_bar
    K = length(fitnesses)
    mean_w = mean_fitness

    var_w = 0.0
    for i in 1:K
        var_w += (fitnesses[i] - mean_w)^2
    end
    var_w /= K

    predicted_delta_w = var_w / mean_w

    return (var_fitness=var_w, predicted_delta_w=predicted_delta_w,
            mean_fitness=mean_w)
end

class ProbabilityDistributionTypes():
    """
    Enum structure to keep hardcoded probability distribution names.
    """
    UNIFORM = "uniform"
    NORMAL = "normal"
    GAMMA = "gamma"
    LOGNORMAL = "lognormal"
    EXPONENTIAL = "exponential"
    BETA = "beta"
    CHISQUARE = "chisquare"
    BINOMIAL = "binomial"
    POISSON = "poisson"
    BERNOULLI = "bernoulli"

    available_distributions = [UNIFORM, NORMAL, GAMMA, LOGNORMAL, EXPONENTIAL, BETA, CHISQUARE, BINOMIAL, POISSON,
                               BERNOULLI]
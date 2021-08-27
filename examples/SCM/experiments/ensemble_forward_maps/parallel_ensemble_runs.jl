@everywhere include("ensemble_run.jl")

# Run ensemble for every var_params entry
@everywhere params = ["entrainment_factor", "detrainment_factor"]
@everywhere val(x::Real) = repeat([x], length(params))
@everywhere var_params = [
    # val(0.0),
    val(0.1),
    val(0.3),
    val(0.5),
    # val(0.8),
    # val(1.0),
]
n_ens=10
@everywhere f_(x) = run_ensemble($params, x, $n_ens)

map(f_, var_params)

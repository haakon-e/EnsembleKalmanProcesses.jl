@everywhere include("ensemble_run.jl")

# Run ensemble for every var_params entry
@everywhere params = ["entr_lognormal_var", "detr_lognormal_var"]
@everywhere val(x::Real) = repeat([x], length(params))
@everywhere var_params = [
    val(0.0),
    val(0.1),
    val(0.3),
    val(0.5),
    val(0.8),
    val(1.0),
    val(5.0),
]
n_ens=20
@everywhere f_(x) = run_ensemble($params, x, $n_ens)

map(f_, var_params)

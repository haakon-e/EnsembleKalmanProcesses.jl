@everywhere include("ensemble_run.jl")

# Run ensemble for every var_params entry
@everywhere var_params = [0.0]
@everywhere f_(x) = run_ensemble(x)

pmap(f_, var_params, 
# on_error=x->nothing
)
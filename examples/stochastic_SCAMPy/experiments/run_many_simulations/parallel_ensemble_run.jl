@everywhere include("ensemble_run.jl")

# Run ensemble for every var_params entry
println("Running parallel jobs")
@everywhere var_params = [0.75]  #[0.0, 0.25, 0.5, 0.75, 1.0]
n_ens=40
@everywhere f_(x) = run_ensemble(x, $n_ens)

pmap(f_, var_params, 
#on_error=x->nothing
)

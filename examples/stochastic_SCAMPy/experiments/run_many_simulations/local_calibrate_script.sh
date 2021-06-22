julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'

# julia --project -p auto ensemble_run.jl
julia --project -p auto parallel_ensemble_run.jl
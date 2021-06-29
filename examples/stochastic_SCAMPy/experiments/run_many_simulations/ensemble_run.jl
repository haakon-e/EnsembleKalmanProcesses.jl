# Import modules to all processes
@everywhere using Pkg
@everywhere Pkg.activate("../..")
@everywhere using Distributions
@everywhere using StatsBase
@everywhere using LinearAlgebra
@everywhere using BlockDiagonals
# Import EKP modules
@everywhere using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
@everywhere using EnsembleKalmanProcesses.Observations
@everywhere using EnsembleKalmanProcesses.ParameterDistributionStorage
@everywhere include(joinpath(@__DIR__, "../../src/helper_funcs.jl"))
using JLD2
using NPZ

"""

sd_param :: standard deviation of stochastic parameter
n_ens :: number of ensembles per EK iteration
"""
function run_ensemble(sd_param, n_ens)
    println("Run ensemble for noise=$sd_param.")
    #########
    #########  User defined parameters and variables
    #########

    # Define the parameters that we want to learn
    param_names = ["entrainment_lognormal_std_dev", "detrainment_lognormal_std_dev"]
    n_param = length(param_names)

    # Define params in the loss function
    loss_params = ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"]
    y_names = Array{String, 1}[]
    push!(y_names, loss_params)
    @assert length(y_names) == 1  # Only one list of variables considered for our 1 simulation.

    # Define directories to look for and store data
    _case = "Bomex"
    les_names = [_case]  # PyCLES case name
    les_suffixes = ["may18"]  # PyCLES case suffix
    les_root = "/groups/esm/ilopezgo" 
    scm_names = ["Stochastic$_case"]  # same as `les_names` in perfect model setting
    scm_data_root = pwd()  # path to folder with `Output.<scm_name>.00000` files

    outdir_root = "/groups/esm/hervik/calibration/output/stochastic_ensembles$(les_names[1])"


    # # Prior information: Define transform to unconstrained gaussian space
    # constraints = [
    #     [bounded(0.5, 2.0)],
    #     ]
    # # All vars are standard Gaussians in unconstrained space
    # prior_dist = [Parameterized(Normal(0.0, 1.0))
    #                 for x in range(1, n_param, length=n_param) ]
    # priors = ParameterDistribution(prior_dist, constraints, param_names)

    ## Set known parameter
    priors = ParameterDistribution(
        repeat([Samples([sd_param])], n_param), 
        repeat([no_constraint()], n_param),
        param_names,
    )

    # Define observation window (s)
    (t_starts, t_ends) = (4.0, 6.0) .* 3600  # 4 to 6 hrs
    # Compute data covariance
    Γy = compute_data_covariance(les_names, les_suffixes, scm_names, y_names, t_starts, t_ends, les_root, scm_data_root)

    #########
    #########  Run ensemble of simulations
    #########

    algo = Inversion() # Sampler(vcat(get_mean(priors)...), get_cov(priors))
    println("NUMBER OF ENSEMBLE MEMBERS: $n_ens")

    initial_params = construct_initial_ensemble(priors, n_ens, rng_seed=rand(1:1000))
    ekobj = EnsembleKalmanProcess(initial_params, yt, Γy, algo)
    scampy_dir = "/groups/esm/hervik/calibration/SCAMPy"  # path to SCAMPy

    # Define caller function
    @everywhere g_(x::Array{Float64,1}) = run_SCAMPy(
            x, $param_names, $scampy_dir, $scm_data_root, $scm_names
        )

    # Create output dir
    outdir_path = joinpath(outdir_root, "noise$(variance_param)")
    println("Name of outdir path for this EKP is: $outdir_path")
    mkpath(outdir_path)

    # Note that the parameters are transformed when used as input to SCAMPy
    params_cons_i = deepcopy(
        transform_unconstrained_to_constrained(
            priors, get_u_final(ekobj)
        )
    )
    params = [row[:] for row in eachrow(params_cons_i')]
    @everywhere params = $params
    ## Run one ensemble forward map (in parallel)
    array_of_tuples = pmap(
        g_, params,
        on_error=ex->nothing,  # ignore errors
        ) # Outer dim is params iterator
    ##
    (sim_dirs_ens,) = ntuple(l->getindex.(array_of_tuples,l),1) # Outer dim is G̃, G 
    sim_dirs_ens_ = filter(x -> !isnothing(x), sim_dirs_ens)  # pmap-error handling

    # get a simulation directory `.../Output.SimName.UUID`, and corresponding parameter name
    for (ens_i, sim_dir) in enumerate(sim_dirs_ens_)  # each ensemble returns a list of simulation directories
        for scm_name in scm_names
            # Copy simulation data to output directory
            dirname = splitpath(sim_dir)[end]
            @assert dirname[1:7] == "Output."  # sanity check
            tmp_data_path = joinpath(sim_dir, "stats/Stats.$scm_name.nc")
            save_data_path = joinpath(outdir_path, "Stats.$scm_name.$ens_i.nc")
            run(`cp $tmp_data_path $save_data_path`)
        end
    end
    println(string("\n\nEKP evaluation 1 finished. \n"))
end

function compute_data_covariance(les_names, les_suffixes, scm_names, y_names, t_starts, t_ends, les_root, scm_data_root)
    @assert (  # Each entry in these lists correspond to one simulation case
        length(les_names) == length(les_suffixes) == length(scm_names) 
        == length(y_names) == length(t_starts) == length(t_ends)
    )
    # Init arrays
    yt = zeros(0)
    yt_var_list = []
    pool_var_list = []  # pooled variance (see `get_time_covariance` in `helper_funcs.jl`)

    for (les_name, les_suffix, scm_name, y_name, tstart, tend) in zip(
            les_names, les_suffixes, scm_names, y_names, t_starts, t_ends
        )
        # Get SCM vertical levels for interpolation
        z_scm = get_profile(joinpath(scm_data_root, "Output.$scm_name.00000"), ["z_half"])
        # Get (interpolated and pool-normalized) observations, get pool variance vector
        les_dir = joinpath(les_root, "Output.$les_name.$les_suffix")
        yt_, yt_var_, pool_var = obs_LES(y_name, les_dir, tstart, tend, z_scm = z_scm)
        push!(pool_var_list, pool_var)
        append!(yt, yt_)
        push!(yt_var_list, yt_var_)
    end
    
    # Construct global observational covariance matrix, TSVD
    Γy = BlockDiagonal(yt_var_list)
    
    return Γy
end

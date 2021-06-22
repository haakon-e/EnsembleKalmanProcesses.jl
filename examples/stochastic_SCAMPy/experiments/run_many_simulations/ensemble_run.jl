# Import modules to all processes
@everywhere using Pkg
@everywhere Pkg.activate("../..")
@everywhere using Distributions
@everywhere using StatsBase
@everywhere using LinearAlgebra
# Import EKP modules
@everywhere using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
@everywhere using EnsembleKalmanProcesses.Observations
@everywhere using EnsembleKalmanProcesses.ParameterDistributionStorage
@everywhere include(joinpath(@__DIR__, "../../src/helper_funcs.jl"))
using JLD2
using NPZ


function run_ensemble(var_param, n_ens)
    #########
    #########  Define the parameters and their priors
    #########
    println("Run ensemble for noise=$var_param.")

    # Define the parameters that we want to learn
    param_names = ["stochastic_noise"]
    n_param = length(param_names)

    # # Prior information: Define transform to unconstrained gaussian space
    # constraints = [
    #     [bounded(0.5, 2.0)],
    #     ]
    # # All vars are standard Gaussians in unconstrained space
    # prior_dist = [Parameterized(Normal(0.0, 1.0))
    #                 for x in range(1, n_param, length=n_param) ]
    # priors = ParameterDistribution(prior_dist, constraints, param_names)

    ## Set known parameter
    variance_param = var_param
    priors = ParameterDistribution(
        [Samples([variance_param])], 
        [no_constraint()], 
        param_names,
    )

    # Define observation window (s)
    t_starts = [4.0] * 3600  # 4hrs
    t_ends = [6.0] * 3600  # 6hrs
    # Define variables considered in the loss function
    y_names = Array{String, 1}[]
    push!(y_names, ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"])
    @assert length(y_names) == 1  # Only one list (prev line) of variables considered for our 1 simulation.

    # Define preconditioning and regularization of inverse problem
    perform_PCA = true # Performs PCA on data

    # Define name of PyCLES simulation to learn from
    les_names = ["Bomex"]
    les_suffixes = ["may18"]
    les_root = "/groups/esm/ilopezgo"
    scm_names = ["StochasticBomex"]  # same as `les_names` in perfect model setting
    scm_data_root = pwd()  # path to folder with `Output.<scm_name>.00000` files

    # Init arrays
    yt = zeros(0)
    yt_var_list = []
    @assert (  # Each entry in these lists correspond to one simulation case
        length(les_names) == length(les_suffixes) == length(scm_names) 
        == length(y_names) == length(t_starts) == length(t_ends)
    )
    for (les_name, les_suffix, scm_name, y_name, tstart, tend) in zip(
            les_names, les_suffixes, scm_names, y_names, t_starts, t_ends
        )
        # Get SCM vertical levels for interpolation
        z_scm = get_profile(joinpath(scm_data_root, "Output.$scm_name.00000"), ["z_half"])
        # Get (interpolated and pool-normalized) observations, get pool variance vector
        les_dir = joinpath(les_root, "Output.$les_name.$les_suffix")
        yt_, yt_var_, pool_var = obs_LES(y_name, les_dir, tstart, tend, z_scm = z_scm)
        if perform_PCA
            yt_pca, yt_var_pca, P_pca = obs_PCA(yt_, yt_var_)
            append!(yt, yt_pca)
            push!(yt_var_list, yt_var_pca)
        else
            append!(yt, yt_)
            push!(yt_var_list, yt_var_)
        end
        # Save full dimensionality (normalized) output for error computation
    end
    d = length(yt) # Length of data array

    # Construct global observational covariance matrix, TSVD
    yt_var = zeros(d, d)
    global vars_num = 1
    for config_cov in yt_var_list
        vars = length(config_cov[1,:])
        yt_var[vars_num:vars_num+vars-1, vars_num:vars_num+vars-1] = config_cov
        global vars_num = vars_num+vars
    end

    Γy = yt_var

    #########
    #########  Run ensemble of simulations
    #########

    algo = Inversion() # Sampler(vcat(get_mean(priors)...), get_cov(priors))
    N_ens = n_ens  # number of ensemble members
    println("NUMBER OF ENSEMBLE MEMBERS: $N_ens")

    initial_params = construct_initial_ensemble(priors, N_ens, rng_seed=rand(1:1000))
    ekobj = EnsembleKalmanProcess(initial_params, yt, Γy, algo)
    scampy_dir = "/groups/esm/hervik/calibration/SCAMPy"  # path to SCAMPy

    # Define caller function
    @everywhere g_(x::Array{Float64,1}) = run_SCAMPy(
            x, $param_names, $scampy_dir, $scm_data_root, $scm_names
        )

    # Create output dir
    outdir_root = "/groups/esm/hervik/calibration/output/fix1"
    outdir_path = joinpath(outdir_root, "noise$(variance_param)")  # joinpath(outdir_root, "results_ensemble_p$(n_param)_e$(N_ens)")
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
    i=1; println(string("\n\nEKP evaluation $i finished. \n"))
end

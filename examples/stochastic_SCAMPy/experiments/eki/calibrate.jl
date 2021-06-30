# This is an example on training the SCAMPy implementation of the EDMF
# scheme with data generated using PyCLES.
#
# The example seeks to find the optimal values of the entrainment and
# detrainment parameters of the EDMF scheme to replicate the LES profiles
# of the BOMEX experiment.
#
# This example is fully parallelized and can be run in the cluster with
# the included script.

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

function run_calibrate()
    #########
    #########  User defined parameters and variables
    #########

    # Define the parameters that we want to learn
    param_names = ["entrainment_lognormal_std_dev", "detrainment_lognormal_std_dev"]
    n_param = length(param_names)

    # Prior information: Define transform to unconstrained gaussian space
    constraints = [[bounded(0.01, 1.0)],
                [bounded(0.01, 1.0)],]
    # All vars are standard Gaussians in unconstrained space
    prior_dist = [Parameterized(Normal(0.5, 1.0)),
                Parameterized(Normal(0.5, 1.0)),]
    priors = ParameterDistribution(prior_dist, constraints, param_names)

    # Define variables considered in the loss function
    loss_params = ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"]
    y_names = Array{String, 1}[]
    push!(y_names, loss_params)
    @assert length(y_names) == 1  # Only one list of variables considered for our 1 simulation.

    # Define name of PyCLES simulations to learn from
    les_names = ["Bomex"]
    les_suffixes = ["may18"]
    les_root = "/groups/esm/ilopezgo"
    scm_names = ["StochasticBomex"]  # same as `les_names` in perfect model setting
    scm_data_root = pwd()  # path to folder with `Output.<scm_name>.00000` files
    scampy_dir = "/groups/esm/hervik/calibration/SCAMPy"  # path to SCAMPy

    outdir_root = "/groups/esm/hervik/calibration/output/eki_bomex"

    # Define observation window (s)
    (t_starts, t_ends) = (4.0, 6.0) .* 3600  # 4 to 6 hrs
    # Compute data covariance
    Γy, pool_var_list, yt = compute_data_covariance(
        les_names, les_suffixes, scm_names, y_names, t_starts, t_ends, les_root, scm_data_root
    )

    #########
    #########  Calibrate: Ensemble Kalman Inversion
    #########

    algo = Inversion() # Sampler(vcat(get_mean(priors)...), get_cov(priors))
    N_ens = 20 # number of ensemble members
    N_iter = 2 # number of EKP iterations.
    Δt = 1.0 # Artificial time stepper of the EKI.
    println("NUMBER OF ENSEMBLE MEMBERS: $N_ens")
    println("NUMBER OF ITERATIONS: $N_iter")

    initial_params = construct_initial_ensemble(priors, N_ens, rng_seed=rand(1:1000))
    ekobj = EnsembleKalmanProcess(initial_params, yt, Γy, algo )

    # Define caller function
    @everywhere g_(x::Array{Float64,1}) = run_SCAMPy(
        x, $param_names, $scampy_dir, 
        $scm_data_root, $scm_names, $y_names, 
        $t_starts, $t_ends, 
        norm_var_list = $pool_var_list,
    )

    # Create output dir
    algo_type = typeof(algo) == Sampler{Float64} ? "eks" : "eki"
    outdir_path = joinpath(outdir_root, "results_$(algo_type)_dt$(Δt)_p$(n_param)_e$(N_ens)_i$(N_iter)_d$d")
    println("Name of outdir path for this EKP is: $outdir_path")
    mkpath(outdir_path)

    # EKP iterations
    g_ens = zeros(N_ens, d)
    norm_err_list = []
    g_big_list = []
    for i in 1:N_iter
        # Note that the parameters are transformed when used as input to SCAMPy
        params_cons_i = deepcopy(
            transform_unconstrained_to_constrained(
                priors, get_u_final(ekobj))
        )
        params = [row[:] for row in eachrow(params_cons_i')]
        @everywhere params = $params
        array_of_tuples = pmap(
            g_, params,
            # on_error=ex->nothing,  # ignore errors
        ) # Outer dim is params iterator
        (sim_dirs_arr, g_ens_arr) = ntuple(l->getindex.(array_of_tuples,l),2) # Outer dim is G̃, G 
        println(string("\n\nEKP evaluation $i finished. Updating ensemble ...\n"))
        for j in 1:N_ens
            g_ens[j, :] = g_ens_arr[j]
          end
        # Get normalized error for full dimensionality output
        push!(norm_err_list, compute_errors(g_ens_arr, yt))
        # Store full dimensionality output
        push!(g_big_list, g_ens_arr)
        # Get normalized error
        if typeof(algo) != Sampler{Float64}
            update_ensemble!(ekobj, Array(g_ens') , Δt_new=Δt)
        else
            update_ensemble!(ekobj, Array(g_ens') )
        end
        println("\nEnsemble updated. Saving results to file...\n")
        # Convert to arrays
        phi_params = Array{Array{Float64,2},1}(transform_unconstrained_to_constrained(priors, get_u(ekobj)))
        phi_params_arr = zeros(i+1, n_param, N_ens)
        for (k,elem) in enumerate(phi_params)
            phi_params_arr[k,:,:] = elem
        end

        # Save EKP information to JLD2 file
        save(string(outdir_path,"/ekp.jld2"),
            "ekp_u", transform_unconstrained_to_constrained(priors, get_u(ekobj)),
            "ekp_g", get_g(ekobj),
            "truth_mean", ekobj.obs_mean,
            "truth_cov", ekobj.obs_noise_cov,
            "ekp_err", ekobj.err,
            "g_big", g_big_list,
            "norm_err", norm_err_list,
            "pool_var", pool_var_list,
            "phi_params", phi_params_arr,
            )
        # Convert to arrays
        phi_params = Array{Array{Float64,2},1}(transform_unconstrained_to_constrained(priors, get_u(ekobj)))
        phi_params_arr = zeros(i+1, n_param, N_ens)
        for (k,elem) in enumerate(phi_params)
            phi_params_arr[k,:,:] = elem
        end
        norm_err_arr = hcat(norm_err_list...)' # N_iter, N_ens
        # Or you can also save information to numpy files with NPZ
        npzwrite(string(outdir_path,"/y_mean.npy"), ekobj.obs_mean)
        npzwrite(string(outdir_path,"/Gamma_y.npy"), ekobj.obs_noise_cov)
        npzwrite(string(outdir_path,"/ekp_err.npy"), ekobj.err)
        npzwrite(string(outdir_path,"/phi_params.npy"), phi_params_arr)
        npzwrite(string(outdir_path,"/norm_err.npy"), norm_err_arr)

        # Save full EDMF data from every ensemble
        eki_iter_path = joinpath(outdir_path, "EKI_iter_$i")
        mkpath(eki_iter_path)
        # get a simulation directory `.../Output.SimName.UUID`, and corresponding parameter name
        for (ens_i, sim_dir) in enumerate(sim_dirs_arr)  # each ensemble returns a list of simulation directories
            for scm_name in scm_names
                # Copy simulation data to output directory
                dirname = splitpath(sim_dir)[end]
                @assert dirname[1:7] == "Output."  # sanity check
                tmp_data_path = joinpath(sim_dir, "stats/Stats.$scm_name.nc")
                save_data_path = joinpath(eki_iter_path, "Stats.$scm_name.$ens_i.nc")
                run(`cp $tmp_data_path $save_data_path`)
            end
        end
    end

    # EKP results: Has the ensemble collapsed toward the truth?
    println("\nEKP ensemble mean at last stage (original space):")
    println( mean( transform_unconstrained_to_constrained(priors, get_u_final(ekobj)), dims=2) ) # Parameters are stored as columns
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
    
    return Γy, pool_var_list, yt
end


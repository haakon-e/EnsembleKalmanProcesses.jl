using JLD2
using PyPlot
using Statistics

# Parameters (edit these lines to plot your own data)
inpath = "/groups/esm/hervik/calibration/EnsembleKalmanProcesses.jl/examples/SCAMPy/experiments/scm_pycles_pipeline/results_eki_dt1.0_p15_e150_i10_d5"         # path to directory of jld2 ekp file
outpath = inpath        # path to directory where output plots should be stored
ekp_path = "ekp.jld2"   # name of ekp data file
param_names = [         # names of parameters that were calibrated
    # mixing length parameters
    "tke_ed_coeff",
    "tke_diss_coeff",
    "static_stab_coeff",
    "lambda_stab",
    # entrainment parameters
    "entrainment_factor",
    "detrainment_factor",
    "turbulent_entrainment_factor",
    "entrainment_smin_tke_coeff",
    "updraft_mixing_frac",
    "entrainment_sigma",
    "sorting_power",
    "aspect_ratio",
    # pressure parameters
    "pressure_normalmode_adv_coeff",     # β₁
    "pressure_normalmode_drag_coeff",    # β₂
    "pressure_normalmode_buoy_coeff1",   # α₁
]

n_param = length(param_names)

# Load data
data = load(joinpath(inpath, ekp_path))

# Mean
phi_m = mean(data["phi_params"], dims=3)[:,:,1]
# Variance
_ustd = std.(data["ekp_u"], dims=2)
n_iter = length(_ustd); n_param = length(_ustd[1])
ustd = zeros((n_iter, n_param))
for i in 1:n_iter ustd[i,:] = _ustd[i] end

# plot parameter evolution
fig, axs = subplots(nrows=n_param, sharex=true, figsize=(15, 4*n_param))
x = 0:n_iter-1
for (i, ax) in enumerate(axs)
    ax.plot(x, phi_m[:,i])
    ax.fill_between(x, 
        phi_m[:,i].-2ustd[:,i], 
        phi_m[:,i].+2ustd[:,i], 
        alpha=0.5,
    )
    ax.set_ylabel(param_names[i])
end

axs[1].set_xlim(0,n_iter-1)
axs[1].set_title("Parameter evolution (mean ±2 std)")
axs[end].set_xlabel("iteration")
savefig(joinpath(outpath, "param_evol.png"))

# Error plot
x = 1:n_iter-1
err = data["ekp_err"]
fig, ax = subplots()
ax.plot(x, err)
ax.set_xlim(1,n_iter-1)
ax.set_ylabel("Error")
ax.set_xlabel("iteration")
ax.set_title("Error evolution")
savefig(joinpath(outpath, "error_evol.png"))


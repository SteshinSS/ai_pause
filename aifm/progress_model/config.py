"""
Configuration file for the progress model.
Contains hardcoded values for numerical stability, caps, and other model parameters.
"""

# =============================================================================
# NUMERICAL STABILITY & PRECISION
# =============================================================================
RHO_COBB_DOUGLAS_THRESHOLD = 1e-9
RHO_LEONTIEF_THRESHOLD = -50.0
SIGMOID_EXPONENT_CLAMP = 100.0
AUTOMATION_FRACTION_CLIP_MIN = 1e-9

# =============================================================================
# PARAMETER CLIPPING (in Parameters.__post_init__)
# =============================================================================
RHO_CLIP_MIN = -50.0
PARAM_CLIP_MIN = 1e-6
AUTOMATION_SLOPE_CLIP_MIN = 0.1
AUTOMATION_SLOPE_CLIP_MAX = 10.0
RESEARCH_STOCK_START_MIN = 1e-10
NORMALIZATION_MIN = 1e-10
experiment_compute_exponent_CLIP_MIN = 0.001
experiment_compute_exponent_CLIP_MAX = 10.0
# Clipping for parallel penalty parameter
PARALLEL_PENALTY_MIN = 0.0
PARALLEL_PENALTY_MAX = 1.0

# =============================================================================
# OTHER RANDOM CONSTANTS
# =============================================================================

# Parallel labor multiplier between average and top for AI2027-SC milestone
# Formula: 30^(1/parallel_penalty) * 30 * PARALLEL_LABOR_MULT_BETWEEN_AVERAGE_AND_TOP_FOR_AI2027_SC
PARALLEL_LABOR_MULT_BETWEEN_AVERAGE_AND_TOP_FOR_AI2027_SC = 25
# Aggregate Research Taste configuration
AGGREGATE_RESEARCH_TASTE_BASELINE = 1.0
AGGREGATE_RESEARCH_TASTE_FALLBACK = 1.0

# Research Taste Distribution Parameters (Log-Normal)
TOP_PERCENTILE = 0.999                    # quantile classed as "top" researchers (0.999 = 99.9th percentile)
MEDIAN_TO_TOP_TASTE_MULTIPLIER = 3.7           # threshold taste ÷ median taste

# Standard deviations for milestone definitions
# Computed from scipy.stats.norm.ppf(TOP_PERCENTILE)
TOP_RESEARCHER_SD = 3.090232306167813     # SD corresponding to TOP_PERCENTILE (99.9th percentile)

# Taste floor interpolation grid configuration
# Below this percentile, floor has negligible effect on mean (return precomputed mean)
TASTE_FLOOR_GRID_LOW_PERCENTILE = 0.001   # 0.1th percentile
# Above this percentile, mean ≈ floor (return floor directly)
TASTE_FLOOR_GRID_HIGH_PERCENTILE = 0.9999  # 99.99th percentile
# Number of grid points for interpolation between low and high percentiles
TASTE_FLOOR_GRID_POINTS = 200

# Research Taste Schedule Configuration (UI-level options)
# Internally, both SD-based options map to 'sd_per_progress' logic; units differ in UI only
TASTE_SCHEDULE_TYPES = ["SDs per effective OOM", "SDs per progress-year"]
DEFAULT_TASTE_SCHEDULE_TYPE = "SDs per progress-year"

# Default AI research taste slope values for different modes
TASTE_SLOPE_DEFAULTS = {
    "SDs per effective OOM": 1.47, # this is out of date since unused
    "SDs per progress-year": 2.1
}

# Horizon Extrapolation Configuration
HORIZON_EXTRAPOLATION_TYPES = ["exponential", "decaying doubling time"]  # Available extrapolation types
DEFAULT_HORIZON_EXTRAPOLATION_TYPE = "decaying doubling time"

# How the AC milestone's progress level is determined (see progress_model/ac_anchor.py)
AC_ANCHOR_MODES = ["time-horizon", "revenue", "eci", "uplift-solved"]
DEFAULT_AC_ANCHOR_MODE = "uplift-solved"  # Q2: the sheet's highest-weight anchor block

# SIE (Software Intelligence Explosion) Mode Configuration
# Available milestones for SIE mode start point
SOS_START_MILESTONES = ["AC", "AI2027-SC", "AIR-5x", "AIR-25x", "AIR-250x", "AIR-2000x", "AIR-10000x"]
DEFAULT_SOS_START_MILESTONE = "AC"

# Manual Horizon Fitting Parameters
#
# These, and the DEFAULT_PARAMETERS entries below, are the daniel-08-07-26 preset --
# the frontend's own default -- so both ends of the stack start from one belief.
#
# NOTE ON THE OLD CONVENTION: present_day used to track "the latest SOTA model" and
# present_horizon its measured p80 (2026.0959 / 69.9 was Claude Opus 4.6). That predates
# per-observable reference dates and is superseded. present_day is now the simulation's
# "now" -- the Q2 uplift vintage -- and a measurement taken on another date binds through
# its own *_reference_date instead of dragging present_day to meet it. 186.0 is METR's
# Claude Mythos Preview p80, measured at 2026.269178, which is why
# horizon_reference_date is set: stating 186 without it would assert a March measurement
# as a July one.
DEFAULT_present_day = 2026.578082  # Q2 2026 uplift vintage (the sheet's "present day")
DEFAULT_present_horizon = 186.0  # Q2: 3.1 h = METR Mythos Preview p80, at the horizon vintage
DEFAULT_present_doubling_time = 0.3328402169209725  # Daniel Q2 median: 2.54/4/6.28 months
DEFAULT_DOUBLING_DIFFICULTY_GROWTH_FACTOR = 0.920  # Will be optimized if None (1 - 0.080)

# AI Research Taste clipping bounds
AI_RESEARCH_TASTE_MIN = 0.0
AI_RESEARCH_TASTE_MAX = 1e30  # Match the upper bound from PARAMETER_BOUNDS
AI_RESEARCH_TASTE_MAX_SD = 10e30

BASE_FOR_SOFTWARE_LOM = 10.0

# labor and compute anchors are based on these
REFERENCE_YEAR = 2024.8
REFERENCE_LABOR_CHANGE = 30.0
REFERENCE_COMPUTE_CHANGE = 0.1

# This is used to set "r". The idea is that we know quickly software efficiency was growing in this year.
SOFTWARE_PROGRESS_SCALE_REFERENCE_YEAR = 2024.0

# Normalize "0 software efficiency" to Grok 3, which was released in 2025.13 and trained with 10^26.54 FLOPs.
# In the continuous training version, the FLOP of the largest training run at time 2025.13 is nailed down by the training H100e time series.
TRAINING_COMPUTE_REFERENCE_YEAR = 2025.13
# TRAINING_COMPUTE_REFERENCE_OOMS = 26.54

# H100-equivalent -> FLOP conversion. Input CSVs store all compute columns in
# H100e (a rate/capacity); the model needs training_compute in OOMs of FLOP for
# EC = TC + SW. Convert with: tc_ooms = log10(training_compute_h100e * FLOP_PER_H100E_YEAR).
# H100 nameplate 1e15 FLOP/s * 0.4 utilization * seconds/year.
FLOP_PER_H100E_YEAR = 0.4e15 * 3.15576e7  # ~1.262e22

# ECI (Epoch Capabilities Index) is anchored at the present day via the
# eci_at_present_day / eci_growth_rate parameters (see ac_anchor.fit_eci_trend);
# the points-per-OOM slope is derived per run from the anchor progress rate.

# benchmarks and gaps mode

# =============================================================================
# MODEL RATE & VALUE CAPS
# =============================================================================
MAX_RESEARCH_EFFORT = 1e30
MAX_NORMALIZED_PROGRESS_RATE = 1e30
TIME_EXTRAPOLATION_WINDOW = 10.0

# =============================================================================
# ODE INTEGRATION
# =============================================================================
PROGRESS_ODE_CLAMP_MAX = 1e30
RESEARCH_STOCK_ODE_CLAMP_MAX = 1e30
ODE_MAX_STEP = 1.0
EULER_FALLBACK_MIN_STEPS = 100
EULER_FALLBACK_STEPS_PER_YEAR = 10
DENSE_OUTPUT_STEP_SIZE = 0.1  # Step size in years for output time series (was 250 points over 50 years = 0.2 years/step)

# ODE step size logging configuration
ODE_STEP_SIZE_LOGGING = False
ODE_SMALL_STEP_THRESHOLD = 1e-6  # Threshold for warning about very small steps
ODE_STEP_VARIATION_THRESHOLD = 100.0  # Threshold for warning about step size variation

# =============================================================================
# PARAMETER ESTIMATION
# =============================================================================
RELATIVE_ERROR_CLIP = 100.0

# Parameter bounds for website sliders and monte carlo sampling
PARAMETER_BOUNDS = {
    'rho_coding_labor': (-10, -.000001),
    'rho_experiment_capacity': (-1, 1),
    'alpha_experiment_capacity': (0.001, 0.999),
    'r_software': (0.1, 10),
    'software_progress_rate_at_reference_year': (0.05, 10),
    'automation_fraction_at_coding_automation_anchor': (0.1, 1.0),
    'swe_multiplier_at_present_day': (1.0, 10.0),
    'automation_logistic_asymptote': (1.0001, 100.0),  # Upper asymptote for logistic schedule
    'coding_labor_normalization': (0.00001, 10),
    'experiment_compute_exponent': (experiment_compute_exponent_CLIP_MIN, experiment_compute_exponent_CLIP_MAX),
    # AI Research Taste parameter bounds
    'ai_research_taste_at_coding_automation_anchor_sd': (-10, 10),
    'ai_research_taste_slope': (0.1, 10.0),
    'progress_at_aa': (1.0, 500),
    'ac_time_horizon_minutes': (960, 100000000000),
    # Revenue AC-anchor mode bounds (annual revenue in $/yr; growth as annual factor)
    'present_revenue': (1e6, 1e14),
    'present_revenue_growth_rate': (1.01, 100.0),
    'revenue_required_for_ac': (1e8, 1e16),
    # ECI AC-anchor mode bounds (ECI score points; growth in points/year)
    'eci_at_present_day': (50.0, 300.0),
    'eci_growth_rate': (1.0, 100.0),
    'eci_required_for_ac': (100.0, 1000.0),
    # 'uplift-solved' AC-anchor mode: the model's OWN endogenous serial coding
    # uplift at a reference date, and the doubling time (anchor-months) of its
    # excess over 1 at that date. Both conditions are solved for against the
    # model's own endogenous uplift, not extrapolated from an exogenous trend.
    'serial_uplift_at_reference_date': (1.01, 1000.0),
    'excess_uplift_doubling_time': (0.6, 240.0),
    'uplift_reference_date': (2024.0, 2035.0),
    # Per-observable elicitation vintages (None => present_day)
    'horizon_reference_date': (2024.0, 2035.0),
    'revenue_reference_date': (2024.0, 2035.0),
    'eci_reference_date': (2024.0, 2035.0),
    # OPTIONAL third condition: solve the asymptote instead of taking it as given
    'serial_uplift_at_ac': (1.01, 1e6),
    # Manual horizon fitting parameter bounds
    'present_day': (2020.0, 2030.0),
    'present_horizon': (0.001, 10000),  # minutes
    'present_doubling_time': (0.01, 2),  # doubling time in present years
    'doubling_difficulty_growth_factor': (0.5, 1.5),  # difficulty growth rate (1 - decay rate)
    # exp capacity pseudoparameters
    'inf_labor_asymptote': (1, 100000),
    'inf_compute_asymptote': (1, 10000000),
    'labor_anchor_exp_cap': (1, 10),
    'compute_anchor_exp_cap': (1, 10),
    'inv_compute_anchor_exp_cap': (1, 10),
    # parallel penalty for experiment capacity CES
    'parallel_penalty': (0.0, 1.0),
    # Research taste distribution parameter
    'median_to_top_taste_multiplier': (1.01, 100.0), # seeing potential errors on the web app if allow up to 500, unclear if they would affect user`:w
    'top_percentile': (0.5, 0.99999),
    'taste_limit': (0, 100),
    'taste_limit_smoothing': (0.001, 0.999),
    # Median-to-best multipliers for general capability milestones
    'strat_ai_m2b': (0.01, 100.0),  # STRAT-AI median-to-bests above SAR
    'ted_ai_m2b': (0.01, 100.0),    # TED-AI median-to-bests above SAR (ASI is always 2 M2Bs above TED-AI)
    # Optimal CES coding labor parameters
    'coding_automation_efficiency_slope': (0.01, 10.0),
    'optimal_ces_eta_init': (1e-12, 1e12),
    'optimal_ces_grid_size': (256, 16384),
    'optimal_ces_frontier_tail_eps': (1e-20, 1e-1),
    'optimal_ces_frontier_cap': (1.0, 1e30),
    'max_serial_coding_labor_multiplier': (1.0, 1e30),
    # coding labor mode (categorical; bounds unused by UI collector)
    'coding_labor_mode': (0, 1),
    # Training compute growth rate parameters
    'constant_training_system_performance_growth_rate': (0.0, 2.0),
    'slowdown_year': (2020.0, 2040.0),
    'post_slowdown_training_system_performance_growth_rate': (0.0, 2.0),
}

# Validation thresholds for parameter combinations
PARAM_VALIDATION_THRESHOLDS = {
    'automation_fraction_superhuman_coder_min': 0.05,
    'automation_fraction_superhuman_coder_max': 1,
    'rho_extreme_abs': 0.8,
    'rho_product_max': 0.5,
    'coding_labor_normalization_max': 10
}

# =============================================================================
# DEFAULT MODEL PARAMETERS (FOR WEBSITE ONLY(? is this true???), MONTE CARLO USES CONFIG/SAMPLING_CONFIG.YAML)
# BH: it is possible that the monte carlo is affected by these, for example if a mparameter is left out of sampling_config.yaml.
# =============================================================================
DEFAULT_PARAMETERS = {
    'rho_coding_labor': -2,
    'direct_input_exp_cap_ces_params': False,
    'r_software': 2.40,
    'software_progress_rate_at_reference_year': 1,
     # coding automation parameters
    'automation_fraction_at_coding_automation_anchor': 1,
    # DERIVED container (see progress_model/parameters.py, "The uplift pair").
    # This is the value the resolver falls back to when NEITHER uplift container
    # is stated; its serial counterpart at the default reference date (= present
    # day, so the growth factor is 1) is exactly 1.8 ** 0.5 = 1.3416407864998738.
    # Stating it explicitly is the supported legacy path.
    'automation_interp_type': "logistic",
    'automation_logistic_asymptote': 1.1,  # Upper asymptote for logistic coding automation schedule (clipped to 1)
    'coding_labor_normalization': 1, # unused
    # AI Research Taste parameters
    'ai_research_taste_at_coding_automation_anchor_sd': 0.0,   # Daniel
    'ai_research_taste_at_coding_automation_anchor_fallback': 1, # Only used if conversion from SD to taste fails
    'ai_research_taste_slope': 3.0,                            # Daniel: sqrt(1 * 9)
    'taste_schedule_type': DEFAULT_TASTE_SCHEDULE_TYPE,
    # Automation anchor params
    'progress_at_aa': None,
    'ac_time_horizon_minutes': 1.2e5,  # Daniel Q2: 1 work-year at 120,000 min/work-year
    'ac_anchor_mode': DEFAULT_AC_ANCHOR_MODE,
    # Revenue AC-anchor mode (used only when ac_anchor_mode == 'revenue'):
    # present-day annual AI revenue ($/yr), its annual growth factor, and the
    # annual revenue level that defines the AC milestone
    'present_revenue': 4.7e10,                # Q2: $0.047T annualized, in DOLLARS
    'present_revenue_growth_rate': 5.0,       # Q2: 2.5/5/10, geomean 5
    'revenue_required_for_ac': 1e12,
    # ECI AC-anchor mode: the (value, rate) pair also defines the capability_ecis
    # metric in every mode; the threshold only matters when ac_anchor_mode == 'eci'.
    'eci_at_present_day': 162.0,
    'eci_growth_rate': 15.0,
    'eci_required_for_ac': 250.0,
    # 'uplift-solved' AC-anchor mode: instead of reading progress_at_aa off an
    # exogenous trend and then solving the lower automation anchor, the logistic
    # schedule's (k, x0) are solved so the model's own endogenous serial uplift
    # hits `serial_uplift_at_reference_date` at `uplift_reference_date` and its
    # excess doubles every `excess_uplift_doubling_time` anchor-months there.
    # progress_at_aa becomes an output (the A(p) = 1 crossing, plus the AC gap).
    # uplift_reference_date = None means present_day.
    #
    # This is the PRIMARY container for the current AI coding uplift, in every
    # mode -- swe_multiplier_at_present_day is a derived view of it. None means
    # "derive it from the legacy present-day parallel belief", DATE-AWARE:
    #   S = 1 + (swe_multiplier_at_present_day ** parallel_penalty - 1) * g,
    #   g = 2 ** ((uplift_reference_date - present_day) * 12 / excess_uplift_doubling_time)
    # which is ~1.34 at defaults (g == 1 there). Resolved at READ time via
    # resolve_serial_uplift_at_reference_date, never materialized on mode
    # switch. Set explicitly to state an independent serial-space belief (the
    # Q2 elicitation is serial @ 2026.578). Setting BOTH containers to
    # inconsistent values raises.
    # The default BELIEF, not a field default: applied by the resolvers only when
    # neither container was stated. See Parameters.serial_uplift_at_reference_date.
    'serial_uplift_at_reference_date': 2.0,   # Daniel Q2: sqrt(0.25 * 4) + 1
    'excess_uplift_doubling_time': 5.0,       # Daniel Q2: sqrt(2.5 * 10) months
    'uplift_reference_date': 2026.578082,     # Q2 uplift vintage
    # Per-observable reference dates: WHEN each anchor observable was measured.
    # None => present_day (bit-identical to the historical single-date behavior).
    # The observable's value AND its companion per-year rate bind at this date
    # on the anchoring mapping, mirroring the uplift_reference_date convention.
    # Load-bearing, not decoration: present_horizon 186.0 is a 2026.269178 measurement
    # and present_revenue a 2026.374429 one, so leaving these None would re-date both to
    # present_day and overstate them by the progress accrued in between.
    'horizon_reference_date': 2026.269178,   # Q2 sheet's time-horizon block
    'revenue_reference_date': 2026.374429,   # Q2 sheet's revenue block
    'eci_reference_date': 2026.524658,       # Q2 sheet's ECI vintage (eci_at_present_day=162)
    # None (default) => automation_logistic_asymptote is an INPUT and (k, x0) are
    # solved. Set it to state how much serial uplift exists when A(p) reaches 1, and
    # the asymptote is solved for too. Does NOT redefine AC -- AC stays the A(p)=1
    # crossing; this only pins the schedule's curvature with an elicitable quantity
    # in place of an unobservable shape parameter.
    'serial_uplift_at_ac': None,
    'automation_anchors': None,
    # Manual horizon fitting parameters
    'present_day': DEFAULT_present_day,
    'present_horizon': DEFAULT_present_horizon,
    'present_doubling_time': DEFAULT_present_doubling_time,
    'doubling_difficulty_growth_factor': DEFAULT_DOUBLING_DIFFICULTY_GROWTH_FACTOR,
    'horizon_extrapolation_type': DEFAULT_HORIZON_EXTRAPOLATION_TYPE,
    # exp throughput parameters
    'inf_labor_asymptote': 15.0,
    'inf_compute_asymptote': 1000,
    'labor_anchor_exp_cap': 1.6, # unused
    'compute_anchor_exp_cap': None, # unused
    'inv_compute_anchor_exp_cap': 2.8,
    'rho_experiment_capacity': -0.155,
    'alpha_experiment_capacity': 0.809,
    'experiment_compute_exponent': 0.655,
    # penalty on parallel coding labor contribution in exp capacity
    'parallel_penalty': 0.5,
    # Gap mode
    'include_gap': 'no gap',
    'gap_years': 0.7,
    # Research taste distribution parameter
    'median_to_top_taste_multiplier': 3.9664793948382653,  # Daniel: sqrt(0.8 * 11) + 1
    'top_percentile': TOP_PERCENTILE,
    # m2b above the best, as a multiplier on absolute taste.
    'taste_limit': 16.0,  # Daniel Q2: sqrt(4 * 64)
    'taste_limit_smoothing': 0.5,
    # Median-to-best multipliers for general capability milestones
    'strat_ai_m2b': 2.0,  # STRAT-AI is 2 M2Bs above SAR (same as SIAR for now)
    # TED-AI median-to-bests above SAR (ASI is always 2 M2Bs above TED-AI)
    'ted_ai_m2b': 1.0001182930033827,  # Daniel: sqrt(0.2673 * 3.742)
    # Coding labor automation (OOMs/progress-year, converted to OOMs/OOM using anchor rate)
    'coding_automation_efficiency_slope': 3.0,
    'optimal_ces_eta_init': 0.05,
    # Optimal CES coding labor configuration
    'coding_labor_mode': 'optimal_ces',  # {'simple_ces','optimal_ces'}
    'optimal_ces_grid_size': 4096,
    'optimal_ces_frontier_tail_eps': 1e-6,
    'optimal_ces_frontier_cap': 1e12,
    'max_serial_coding_labor_multiplier': 1e8,
    # SOS mode
    'sos_mode': False,
    'sos_start_milestone': DEFAULT_SOS_START_MILESTONE,
    # Training compute growth rate parameters (from sampling_config_brendan.yaml)
    'constant_training_system_performance_growth_rate': 0.6,
    'slowdown_year': 2028.0,
    'post_slowdown_training_system_performance_growth_rate': 0.25,
}

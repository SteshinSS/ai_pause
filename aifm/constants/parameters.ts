
export type ParameterPrimitive = number | string | boolean | null;

export interface ParametersType {
    [key: string]: ParameterPrimitive;
    taste_schedule_type: string;
    present_doubling_time: number;
    ac_time_horizon_minutes: number;
    doubling_difficulty_growth_factor: number;
    rho_coding_labor: number;
    rho_experiment_capacity: number;
    alpha_experiment_capacity: number;
    direct_input_exp_cap_ces_params: boolean;
    r_software: number;
    software_progress_rate_at_reference_year: number;
    coding_labor_normalization: number;
    experiment_compute_exponent: number;
    coding_labor_exponent: number;
    automation_fraction_at_coding_automation_anchor: number;
    automation_interp_type: string;
    automation_logistic_asymptote: number;
    // LEGACY container for the current AI coding uplift (parallel space, at
    // present_day). serial_uplift_at_reference_date is the source of truth;
    // presets that state the serial container leave this null.
    swe_multiplier_at_present_day: number | null;
    automation_anchors: ParameterPrimitive;
    ai_research_taste_at_coding_automation_anchor_sd: number;
    ai_research_taste_slope: number;
    progress_at_aa: number;
    present_day: number;
    present_horizon: number;
    horizon_extrapolation_type: string;
    ac_anchor_mode: string;
    // Per-observable elicitation vintages: when set, the observable's value
    // and its per-year rate bind at this date instead of present_day.
    horizon_reference_date: number | null;
    revenue_reference_date: number | null;
    eci_reference_date: number | null;
    present_revenue: number;
    present_revenue_growth_rate: number;
    revenue_required_for_ac: number;
    serial_uplift_at_reference_date: number | null;
    excess_uplift_doubling_time: number;
    uplift_reference_date: number | null;
    serial_uplift_at_ac: number | null;
    eci_at_present_day: number;
    eci_growth_rate: number;
    eci_required_for_ac: number;
    inf_labor_asymptote: number;
    inf_compute_asymptote: number;
    labor_anchor_exp_cap: number;
    compute_anchor_exp_cap: ParameterPrimitive;
    inv_compute_anchor_exp_cap: number;
    benchmarks_and_gaps_mode: boolean;
    gap_years: number;
    coding_automation_efficiency_slope: number;
    max_serial_coding_labor_multiplier: number;
    median_to_top_taste_multiplier: number;
    top_percentile: number;
    taste_limit: number;
    taste_limit_smoothing: number;
    strat_ai_m2b: number;
    ted_ai_m2b: number;
    optimal_ces_eta_init: number;
    constant_training_system_performance_growth_rate: number;
    slowdown_year: number;
    post_slowdown_training_system_performance_growth_rate: number;
}

// --- Frozen forecaster snapshots ---
// Each snapshot is a complete, hardcoded ParametersType. No snapshot references
// dynamic config or any other snapshot, so URLs that encode diffs against them are
// stable forever regardless of future default changes.

const FROZEN_PRESETS: Record<string, ParametersType> = {
  // Eli's median (original Python defaults at the time, present_day=2025.9, present_horizon=32)
  'eli-12-29-25': {
    taste_schedule_type: 'SDs per progress-year',
    present_doubling_time: 0.458,
    ac_time_horizon_minutes: 7.176091259055681,
    doubling_difficulty_growth_factor: 0.92,
    rho_coding_labor: -2.0,
    rho_experiment_capacity: -0.155,
    alpha_experiment_capacity: 0.809,
    direct_input_exp_cap_ces_params: false,
    r_software: 2.4,
    software_progress_rate_at_reference_year: 1.0,
    coding_labor_normalization: 1.0,
    experiment_compute_exponent: 0.655,
    coding_labor_exponent: 0.5,
    automation_fraction_at_coding_automation_anchor: 1.0,
    automation_interp_type: 'logistic',
    automation_logistic_asymptote: 1.1,
    swe_multiplier_at_present_day: null,                        // legacy parallel container: not stated
    automation_anchors: null,
    ai_research_taste_at_coding_automation_anchor_sd: 0.5,
    ai_research_taste_slope: 2.1,
    progress_at_aa: 1.0,
    present_day: 2025.9,
    present_horizon: 32.0,
    horizon_extrapolation_type: 'decaying doubling time',
    ac_anchor_mode: 'time-horizon',
    horizon_reference_date: null,
    revenue_reference_date: null,
    eci_reference_date: null,
    present_revenue: 1e10,
    present_revenue_growth_rate: 3.0,
    revenue_required_for_ac: 1e12,
    serial_uplift_at_reference_date: 1.3416407864998738,        // former swe value ** coding_labor_exponent (0.5)
    excess_uplift_doubling_time: 12.0,
    uplift_reference_date: null,
    serial_uplift_at_ac: null,
    eci_at_present_day: 162.0,
    eci_growth_rate: 15.0,
    eci_required_for_ac: 250.0,
    inf_labor_asymptote: 15.0,
    inf_compute_asymptote: 1000.0,
    labor_anchor_exp_cap: 1.6,
    compute_anchor_exp_cap: null,
    inv_compute_anchor_exp_cap: 2.8,
    benchmarks_and_gaps_mode: false,
    gap_years: 0.7,
    coding_automation_efficiency_slope: 3.0,
    max_serial_coding_labor_multiplier: 100000000.0,
    median_to_top_taste_multiplier: 3.7,
    top_percentile: 0.999,
    taste_limit: 8.0,
    taste_limit_smoothing: 0.5,
    strat_ai_m2b: 2.0,
    ted_ai_m2b: 1.5,
    optimal_ces_eta_init: 0.05,
    constant_training_system_performance_growth_rate: 0.6,
    slowdown_year: 2028.0,
    post_slowdown_training_system_performance_growth_rate: 0.25,
  } as ParametersType,

  // Daniel's median (01/26/26): eli-12-29-25 base + Daniel's 5 median overrides
  // present_day=2025.9, present_horizon=32 (same as eli snapshot at the time)
  'daniel-12-29-25': {
    taste_schedule_type: 'SDs per progress-year',
    present_doubling_time: 0.45599342100517193,   // Daniel: sqrt(0.29 * 0.717)
    ac_time_horizon_minutes: 5.595378599560064,   // Daniel: log10(sqrt(24912 * 6228000))
    doubling_difficulty_growth_factor: 0.92,
    rho_coding_labor: -2.0,
    rho_experiment_capacity: -0.155,
    alpha_experiment_capacity: 0.809,
    direct_input_exp_cap_ces_params: false,
    r_software: 2.4,
    software_progress_rate_at_reference_year: 1.0,
    coding_labor_normalization: 1.0,
    experiment_compute_exponent: 0.655,
    coding_labor_exponent: 0.5,
    automation_fraction_at_coding_automation_anchor: 1.0,
    automation_interp_type: 'logistic',
    automation_logistic_asymptote: 1.1,
    swe_multiplier_at_present_day: null,                        // legacy parallel container: not stated
    automation_anchors: null,
    ai_research_taste_at_coding_automation_anchor_sd: 0,   // Daniel
    ai_research_taste_slope: 3.0,                          // Daniel: sqrt(1 * 9)
    progress_at_aa: 1.0,
    present_day: 2025.9,
    present_horizon: 32.0,
    horizon_extrapolation_type: 'decaying doubling time',
    ac_anchor_mode: 'time-horizon',
    horizon_reference_date: null,
    revenue_reference_date: null,
    eci_reference_date: null,
    present_revenue: 1e10,
    present_revenue_growth_rate: 3.0,
    revenue_required_for_ac: 1e12,
    serial_uplift_at_reference_date: 1.3416407864998738,        // former swe value ** coding_labor_exponent (0.5)
    excess_uplift_doubling_time: 12.0,
    uplift_reference_date: null,
    serial_uplift_at_ac: null,
    eci_at_present_day: 162.0,
    eci_growth_rate: 15.0,
    eci_required_for_ac: 250.0,
    inf_labor_asymptote: 15.0,
    inf_compute_asymptote: 1000.0,
    labor_anchor_exp_cap: 1.6,
    compute_anchor_exp_cap: null,
    inv_compute_anchor_exp_cap: 2.8,
    benchmarks_and_gaps_mode: false,
    gap_years: 0.7,
    coding_automation_efficiency_slope: 3.0,
    max_serial_coding_labor_multiplier: 100000000.0,
    median_to_top_taste_multiplier: 3.9664793948382653,   // Daniel: sqrt(0.8 * 11) + 1
    top_percentile: 0.999,
    taste_limit: 8.0,
    taste_limit_smoothing: 0.5,
    strat_ai_m2b: 2.0,
    ted_ai_m2b: 1.5,
    optimal_ces_eta_init: 0.05,
    constant_training_system_performance_growth_rate: 0.6,
    slowdown_year: 2028.0,
    post_slowdown_training_system_performance_growth_rate: 0.25,
  } as ParametersType,

  // Eli's median (04/02/26): updated medians from current sampling_config.yaml
  'eli-04-02-26': {
    taste_schedule_type: 'SDs per progress-year',
    present_doubling_time: 0.37488146126475763,                   // sqrt(0.1833 * 0.7667)
    ac_time_horizon_minutes: 7.176091259055681,                 // log10(sqrt(12000 * 1.875e10))
    doubling_difficulty_growth_factor: 0.92,
    rho_coding_labor: -2.0,
    rho_experiment_capacity: -0.155,
    alpha_experiment_capacity: 0.809,
    direct_input_exp_cap_ces_params: false,
    r_software: 2.4,
    software_progress_rate_at_reference_year: 1.0,
    coding_labor_normalization: 1.0,
    experiment_compute_exponent: 0.655,
    coding_labor_exponent: 0.5,
    automation_fraction_at_coding_automation_anchor: 1.0,
    automation_interp_type: 'logistic',
    automation_logistic_asymptote: 1.1,                         // sqrt(0.005 * 2) + 1
    swe_multiplier_at_present_day: null,                        // legacy parallel container: not stated
    automation_anchors: null,
    ai_research_taste_at_coding_automation_anchor_sd: 0.5,      // (-2.5 + 3.5) / 2
    ai_research_taste_slope: 2.1,                               // sqrt(0.7 * 6.3)
    progress_at_aa: 1.0,
    present_day: 2026.0959,                                     // Updated model anchor
    present_horizon: 69.9,                                      // Updated model anchor
    horizon_extrapolation_type: 'decaying doubling time',
    ac_anchor_mode: 'time-horizon',
    horizon_reference_date: null,
    revenue_reference_date: null,
    eci_reference_date: null,
    present_revenue: 1e10,
    present_revenue_growth_rate: 3.0,
    revenue_required_for_ac: 1e12,
    serial_uplift_at_reference_date: 1.4142135623730951,        // former swe value ** coding_labor_exponent (0.5)
    excess_uplift_doubling_time: 12.0,
    uplift_reference_date: null,
    serial_uplift_at_ac: null,
    eci_at_present_day: 162.0,
    eci_growth_rate: 15.0,
    eci_required_for_ac: 250.0,
    inf_labor_asymptote: 15.142135623730951,                    // sqrt(1 * 200) + 1
    inf_compute_asymptote: 1001.0,                              // sqrt(25 * 40000) + 1
    labor_anchor_exp_cap: 1.6,
    compute_anchor_exp_cap: null,
    inv_compute_anchor_exp_cap: 2.794435844492636,              // sqrt(0.7 * 4.6) + 1
    benchmarks_and_gaps_mode: false,
    gap_years: 0.7,                                             // sqrt(0.1 * 4.9)
    coding_automation_efficiency_slope: 3.0,                    // sqrt(1 * 9)
    max_serial_coding_labor_multiplier: 100000000.0,            // sqrt(1e4 * 1e12)
    median_to_top_taste_multiplier: 3.698147512646408,          // sqrt(0.7 * 10.4) + 1
    top_percentile: 0.999,
    taste_limit: 8.0,                                           // sqrt(2 * 32)
    taste_limit_smoothing: 0.5,
    strat_ai_m2b: 2.0,
    ted_ai_m2b: 1.4966629547095764,                             // sqrt(0.4 * 5.6)
    optimal_ces_eta_init: 0.05,
    constant_training_system_performance_growth_rate: 0.6,
    slowdown_year: 2028.0,
    post_slowdown_training_system_performance_growth_rate: 0.25,
  } as ParametersType,

  // Daniel's median (04/02/26): updated medians from current sampling configs
  'daniel-04-02-26': {
    taste_schedule_type: 'SDs per progress-year',
    present_doubling_time: 0.3328402169209725,   // Daniel: sqrt(0.2117 * 0.5233)
    ac_time_horizon_minutes: 5.095378599560064,   // Daniel: log10(sqrt(6228 * 2491200))
    doubling_difficulty_growth_factor: 0.92,
    rho_coding_labor: -2.0,
    rho_experiment_capacity: -0.155,
    alpha_experiment_capacity: 0.809,
    direct_input_exp_cap_ces_params: false,
    r_software: 2.4,
    software_progress_rate_at_reference_year: 1.0,
    coding_labor_normalization: 1.0,
    experiment_compute_exponent: 0.655,
    coding_labor_exponent: 0.5,
    automation_fraction_at_coding_automation_anchor: 1.0,
    automation_interp_type: 'logistic',
    automation_logistic_asymptote: 1.1,
    swe_multiplier_at_present_day: null,                        // legacy parallel container: not stated
    automation_anchors: null,
    ai_research_taste_at_coding_automation_anchor_sd: 0,   // Daniel
    ai_research_taste_slope: 3.0,                          // Daniel: sqrt(1 * 9)
    progress_at_aa: 1.0,
    present_day: 2026.0959,                                // Updated model anchor
    present_horizon: 69.9,                                 // Updated model anchor
    horizon_extrapolation_type: 'decaying doubling time',
    ac_anchor_mode: 'time-horizon',
    horizon_reference_date: null,
    revenue_reference_date: null,
    eci_reference_date: null,
    present_revenue: 1e10,
    present_revenue_growth_rate: 3.0,
    revenue_required_for_ac: 1e12,
    serial_uplift_at_reference_date: 1.4142135623730951,        // former swe value ** coding_labor_exponent (0.5)
    excess_uplift_doubling_time: 12.0,
    uplift_reference_date: null,
    serial_uplift_at_ac: null,
    eci_at_present_day: 162.0,
    eci_growth_rate: 15.0,
    eci_required_for_ac: 250.0,
    inf_labor_asymptote: 15.0,
    inf_compute_asymptote: 1000.0,
    labor_anchor_exp_cap: 1.6,
    compute_anchor_exp_cap: null,
    inv_compute_anchor_exp_cap: 2.8,
    benchmarks_and_gaps_mode: false,
    gap_years: 0.7,
    coding_automation_efficiency_slope: 3.0,
    max_serial_coding_labor_multiplier: 100000000.0,
    median_to_top_taste_multiplier: 3.9664793948382653,   // Daniel: sqrt(0.8 * 11) + 1
    top_percentile: 0.999,
    taste_limit: 16.0,                                    // Daniel: sqrt(4 * 64)
    taste_limit_smoothing: 0.5,
    strat_ai_m2b: 2.0,
    ted_ai_m2b: 1.0001182930033827,                       // Daniel: sqrt(0.2673 * 3.742)
    optimal_ces_eta_init: 0.05,
    constant_training_system_performance_growth_rate: 0.6,
    slowdown_year: 2028.0,
    post_slowdown_training_system_performance_growth_rate: 0.25,
  } as ParametersType,

  // Eli's median (08/10/26; renamed from eli-08-07-26, values unchanged —
  // the 08/10 sheet revision moved only the revenue-growth tails, median intact):
  // eli-04-02-26 base + Eli's FULL Q2 2026 sheet
  // columns (config/sampling_config_q2_eli.yaml): uplift block plus the
  // time-horizon and revenue blocks at their own vintages, so switching the
  // AC anchor mode dropdown anchors on Eli's Q2 numbers. Solved-uplift mode:
  // the automation schedule is solved from the uplift level and doubling time,
  // so ac_time_horizon_minutes is carried along but has no effect on AC.
  // Uplift-at-AC (Eli 4/20/121.33) is elicited but NOT set, matching the
  // sampling config: the asymptote stays a direct input.
  'eli-08-16-26': {
    taste_schedule_type: 'SDs per progress-year',
    present_doubling_time: 0.37488146126475763,                 // Eli: sqrt(0.1833 * 0.7667) — unchanged from his standing belief (2.2/4.5/9.2 mo); the Q2 sheet's 2.4/-/9.6 cells were formula artifacts
    ac_time_horizon_minutes: 7.176091259055681,                 // Eli Q2: 125 work-years, log10(sqrt(12000 * 1.875e10))
    doubling_difficulty_growth_factor: 0.92,
    rho_coding_labor: -2.0,
    rho_experiment_capacity: -0.155,
    alpha_experiment_capacity: 0.809,
    direct_input_exp_cap_ces_params: false,
    r_software: 2.4,
    software_progress_rate_at_reference_year: 1.0,
    coding_labor_normalization: 1.0,
    experiment_compute_exponent: 0.655,
    coding_labor_exponent: 0.5,
    automation_fraction_at_coding_automation_anchor: 1.0,
    automation_interp_type: 'logistic',
    automation_logistic_asymptote: 1.1,                         // sqrt(0.005 * 2) + 1
    swe_multiplier_at_present_day: null,                        // not stated: the serial container below is the source of truth
    automation_anchors: null,
    ai_research_taste_at_coding_automation_anchor_sd: 0.5,      // (-2.5 + 3.5) / 2
    ai_research_taste_slope: 2.1,                               // sqrt(0.7 * 6.3)
    progress_at_aa: 1.0,
    present_day: 2026.578082,                                   // = the Q2 uplift vintage, so the anchor and the uplift belief share one date
    present_horizon: 186.0,                                     // Q2: 3.1 hours at the horizon vintage
    horizon_extrapolation_type: 'decaying doubling time',
    ac_anchor_mode: 'uplift-solved',                            // Q2 uplift block
    // Each observable binds at its Q2 sheet vintage ("Present day" row of its
    // block). ECI has no Q2 block of its own, but eci_at_present_day = 162 is a
    // late-July claim and cannot bind at the old Q1 date: Epoch's frontier read
    // 155.7 in early February and has sat at 161.65 since 2026-07-09, so 162 is
    // only true at the sheet's ECI vintage. This also closes the gap with the
    // sampling configs, which leave eci_reference_date unset => present_day
    // (2026.578) -- the same frontier plateau, ~0.08 OOM away.
    horizon_reference_date: 2026.269178,
    revenue_reference_date: 2026.374429,
    eci_reference_date: 2026.524658,
    present_revenue: 4.7e10,                                    // Q2: $0.047T annualized
    present_revenue_growth_rate: 5.0,                           // Q2 (rev. 08/10/26): 2/5/12.5, geomean 5
    revenue_required_for_ac: 7e12,                              // Q2: 0.2/7/245 T, geomean $7T
    serial_uplift_at_reference_date: 1.7,                       // Eli: sqrt(0.1 * 4.9) + 1
    excess_uplift_doubling_time: 6.500769185258003,             // Eli: sqrt(2 * 21.13), elicited median 6.5
    uplift_reference_date: 2026.578082,                         // Q2 uplift vintage
    serial_uplift_at_ac: null,
    eci_at_present_day: 162.0,
    eci_growth_rate: 15.0,
    eci_required_for_ac: 250.0,
    inf_labor_asymptote: 15.142135623730951,                    // sqrt(1 * 200) + 1
    inf_compute_asymptote: 1001.0,                              // sqrt(25 * 40000) + 1
    labor_anchor_exp_cap: 1.6,
    compute_anchor_exp_cap: null,
    inv_compute_anchor_exp_cap: 2.794435844492636,              // sqrt(0.7 * 4.6) + 1
    benchmarks_and_gaps_mode: false,
    gap_years: 0.7,                                             // sqrt(0.1 * 4.9)
    coding_automation_efficiency_slope: 3.0,                    // sqrt(1 * 9)
    max_serial_coding_labor_multiplier: 100000000.0,            // sqrt(1e4 * 1e12)
    median_to_top_taste_multiplier: 3.698147512646408,          // sqrt(0.7 * 10.4) + 1
    top_percentile: 0.999,
    taste_limit: 8.0,                                           // sqrt(2 * 32)
    taste_limit_smoothing: 0.5,
    strat_ai_m2b: 2.0,
    ted_ai_m2b: 1.4966629547095764,                             // sqrt(0.4 * 5.6)
    optimal_ces_eta_init: 0.05,
    constant_training_system_performance_growth_rate: 0.6,
    slowdown_year: 2028.0,
    post_slowdown_training_system_performance_growth_rate: 0.25,
  } as ParametersType,

  // Daniel's median (08/10/26; renamed from daniel-08-07-26, values unchanged):
  // daniel-04-02-26 base + Daniel's FULL Q2 2026
  // sheet columns (config/sampling_config_q2_daniel.yaml): uplift block plus
  // the time-horizon and revenue blocks at their own vintages, so switching
  // the AC anchor mode dropdown anchors on Daniel's Q2 numbers, not stale
  // base values. Uplift-at-AC (Daniel 4/20/121.33) not set.
  'daniel-08-16-26': {
    taste_schedule_type: 'SDs per progress-year',
    present_doubling_time: 0.3328402169209725,   // Daniel Q2: 2.54/4/6.28 months, sqrt(0.2117 * 0.5233) yr
    ac_time_horizon_minutes: 5.079181246047625,   // Daniel Q2: 1 work-year, log10(120000 min)
    doubling_difficulty_growth_factor: 0.92,
    rho_coding_labor: -2.0,
    rho_experiment_capacity: -0.155,
    alpha_experiment_capacity: 0.809,
    direct_input_exp_cap_ces_params: false,
    r_software: 2.4,
    software_progress_rate_at_reference_year: 1.0,
    coding_labor_normalization: 1.0,
    experiment_compute_exponent: 0.655,
    coding_labor_exponent: 0.5,
    automation_fraction_at_coding_automation_anchor: 1.0,
    automation_interp_type: 'logistic',
    automation_logistic_asymptote: 1.1,
    swe_multiplier_at_present_day: null,                   // not stated: the serial container below is the source of truth
    automation_anchors: null,
    ai_research_taste_at_coding_automation_anchor_sd: 0,   // Daniel
    ai_research_taste_slope: 3.0,                          // Daniel: sqrt(1 * 9)
    progress_at_aa: 1.0,
    present_day: 2026.578082,                              // = the Q2 uplift vintage, so the anchor and the uplift belief share one date
    present_horizon: 186.0,                                // Q2: 3.1 hours at the horizon vintage
    horizon_extrapolation_type: 'decaying doubling time',
    ac_anchor_mode: 'uplift-solved',                       // Q2 uplift block
    // Each observable binds at its Q2 sheet vintage ("Present day" row of its
    // block). ECI has no Q2 block of its own, but eci_at_present_day = 162 is a
    // late-July claim and cannot bind at the old Q1 date: Epoch's frontier read
    // 155.7 in early February and has sat at 161.65 since 2026-07-09, so 162 is
    // only true at the sheet's ECI vintage. This also closes the gap with the
    // sampling configs, which leave eci_reference_date unset => present_day
    // (2026.578) -- the same frontier plateau, ~0.08 OOM away.
    horizon_reference_date: 2026.269178,
    revenue_reference_date: 2026.374429,
    eci_reference_date: 2026.524658,
    present_revenue: 4.7e10,                               // Q2: $0.047T annualized
    present_revenue_growth_rate: 5.0,                      // Q2 (rev. 08/10/26): 2/5/12.5, geomean 5
    revenue_required_for_ac: 1e12,                         // Q2: 0.1/1/10 T, geomean $1T
    serial_uplift_at_reference_date: 2.0,                  // Daniel: sqrt(0.25 * 4) + 1
    excess_uplift_doubling_time: 5.0,                      // Daniel: sqrt(2.5 * 10)
    uplift_reference_date: 2026.578082,                    // Q2 uplift vintage
    serial_uplift_at_ac: null,
    eci_at_present_day: 162.0,
    eci_growth_rate: 15.0,
    eci_required_for_ac: 250.0,
    inf_labor_asymptote: 15.0,
    inf_compute_asymptote: 1000.0,
    labor_anchor_exp_cap: 1.6,
    compute_anchor_exp_cap: null,
    inv_compute_anchor_exp_cap: 2.794435844492636,   // sqrt(0.7 * 4.6) + 1 (was rounded 2.8)
    benchmarks_and_gaps_mode: false,
    gap_years: 0.7,
    coding_automation_efficiency_slope: 3.0,
    max_serial_coding_labor_multiplier: 100000000.0,
    median_to_top_taste_multiplier: 3.9664793948382653,   // Daniel: sqrt(0.8 * 11) + 1
    top_percentile: 0.999,
    taste_limit: 16.0,                                    // Daniel: sqrt(4 * 64)
    taste_limit_smoothing: 0.5,
    strat_ai_m2b: 2.0,
    ted_ai_m2b: 1.0001182930033827,                       // Daniel: sqrt(0.2673 * 3.742)
    optimal_ces_eta_init: 0.05,
    constant_training_system_performance_growth_rate: 0.6,
    slowdown_year: 2028.0,
    post_slowdown_training_system_performance_growth_rate: 0.25,
  } as ParametersType,

  // Brendan's median (08/10/26): eli-08-10-26 base (Brendan inherits the
  // shared base config for everything not stated in
  // config/sampling_config_q2_brendan.yaml) + Brendan's Q2 uplift and revenue
  // blocks, and the TastEval-conditioned taste block (commit 31951704:
  // reasoning-era ECM trend, hierarchical error bars, shape-only likelihood).
  // Brendan's mixture is uplift-solved 0.65 / revenue 0.35; the preset pins
  // the dominant mode. Time-horizon cells are blank in his sheet (weight 0);
  // the horizon rows below are the inherited base beliefs.
  'brendan-08-16-26': {
    taste_schedule_type: 'SDs per progress-year',
    present_doubling_time: 0.37488146126475763,                 // inherited base: sqrt(0.1833 * 0.7667)
    ac_time_horizon_minutes: 7.176091259055681,                 // inherited base row (125 work-years); no effect in uplift-solved mode
    doubling_difficulty_growth_factor: 0.92,
    rho_coding_labor: -2.0,
    rho_experiment_capacity: -0.155,
    alpha_experiment_capacity: 0.809,
    direct_input_exp_cap_ces_params: false,
    r_software: 2.4,
    software_progress_rate_at_reference_year: 1.0,
    coding_labor_normalization: 1.0,
    experiment_compute_exponent: 0.655,
    coding_labor_exponent: 0.5,
    automation_fraction_at_coding_automation_anchor: 1.0,
    automation_interp_type: 'logistic',
    automation_logistic_asymptote: 1.1,                         // sqrt(0.005 * 2) + 1
    swe_multiplier_at_present_day: null,                        // not stated: the serial container below is the source of truth
    automation_anchors: null,
    ai_research_taste_at_coding_automation_anchor_sd: 0.5,      // (-2.5 + 3.5) / 2
    ai_research_taste_slope: 2.692582403567252,                 // Brendan: sqrt(1.25 * 5.8) — TastEval-conditioned ci80
    progress_at_aa: 1.0,
    present_day: 2026.578082,                                   // = the Q2 uplift vintage, so the anchor and the uplift belief share one date
    present_horizon: 186.0,                                     // Q2: 3.1 hours at the horizon vintage
    horizon_extrapolation_type: 'decaying doubling time',
    ac_anchor_mode: 'uplift-solved',                            // Brendan's dominant mode (0.65 weight)
    horizon_reference_date: 2026.269178,
    revenue_reference_date: 2026.374429,
    eci_reference_date: 2026.524658,
    present_revenue: 4.7e10,                                    // Q2: $0.047T annualized
    present_revenue_growth_rate: 7.0,                           // Brendan Q2: 3.5/7/14, geomean 7
    revenue_required_for_ac: 3e12,                              // Brendan Q2: 0.3/3/30 T, geomean $3T
    serial_uplift_at_reference_date: 1.8,                       // Brendan: sqrt(0.16 * 4.0) + 1
    excess_uplift_doubling_time: 5.499090833947008,             // Brendan: sqrt(1.89 * 16), elicited median 5.5
    uplift_reference_date: 2026.578082,                         // Q2 uplift vintage
    serial_uplift_at_ac: null,
    eci_at_present_day: 162.0,
    eci_growth_rate: 15.0,
    eci_required_for_ac: 250.0,
    inf_labor_asymptote: 15.142135623730951,                    // sqrt(1 * 200) + 1
    inf_compute_asymptote: 1001.0,                              // sqrt(25 * 40000) + 1
    labor_anchor_exp_cap: 1.6,
    compute_anchor_exp_cap: null,
    inv_compute_anchor_exp_cap: 2.794435844492636,              // sqrt(0.7 * 4.6) + 1
    benchmarks_and_gaps_mode: false,
    gap_years: 0.7,                                             // sqrt(0.1 * 4.9)
    coding_automation_efficiency_slope: 3.0,                    // sqrt(1 * 9)
    max_serial_coding_labor_multiplier: 100000000.0,            // sqrt(1e4 * 1e12)
    median_to_top_taste_multiplier: 4.346640106136302,          // Brendan: sqrt(1.0 * 11.2) + 1 — TastEval-conditioned ci80
    top_percentile: 0.999,
    taste_limit: 8.0,                                           // sqrt(2 * 32)
    taste_limit_smoothing: 0.5,
    strat_ai_m2b: 2.0,
    ted_ai_m2b: 1.4966629547095764,                             // sqrt(0.4 * 5.6)
    optimal_ces_eta_init: 0.05,
    constant_training_system_performance_growth_rate: 0.6,
    slowdown_year: 2028.0,
    post_slowdown_training_system_performance_growth_rate: 0.25,
  } as ParametersType,
};

// --- Current default ---
// Points to the latest Daniel preset. DEFAULT_PARAMETERS consumers (components,
// server API, etc.) continue to work unchanged.
export const CURRENT_DEFAULT_BASELINE_ID = 'daniel-08-16-26';
// `uplift_reference_date` is null by default, meaning "use present_day". That
// convention has to be applied at every read site or the value silently falls
// back to whatever the consumer happens to do with null (a slider clamps to its
// minimum, a chart may use present_day) -- which is exactly how the slider and
// the chart ended up disagreeing. Resolve through here instead of reading the
// raw field.
export const resolveParamValue = (
    paramName: string,
    params: ParametersType,
): number => {
    const raw = (params as unknown as Record<string, unknown>)[paramName];
    if (raw == null) {
        // Reference dates (uplift, horizon, revenue, ECI): null => the present day.
        if (paramName === 'uplift_reference_date'
            || paramName === 'horizon_reference_date'
            || paramName === 'revenue_reference_date'
            || paramName === 'eci_reference_date') {
            return Number(params.present_day);
        }
    }
    return raw as number;
};

export const DEFAULT_PARAMETERS: ParametersType = FROZEN_PRESETS['daniel-08-16-26'];

// Training FLOP of the reference model (Grok 3), at progress_model/config.py's
// TRAINING_COMPUTE_REFERENCE_YEAR = 2025.13. Used as the effective-compute ->
// progress offset: progress = effectiveCompute_OOM - this.
//
// Python no longer declares it (progress_model/config.py:138 is commented out). Instead the
// H100e input series is built so the CUMULATIVE HARDWARE-FLOP stock reaches this value at
// the reference year (training_compute, not effective_compute, which is offset by the
// software-efficiency baseline) -- see docs/input-series-provenance.md for the derivation.
// Keep in step with that by hand.
export const TRAINING_COMPUTE_REFERENCE_OOMS = 26.54;

export function areParametersAtDefaults(parameters: ParametersType): boolean {
    return Object.keys(DEFAULT_PARAMETERS).every(key => {
        const paramKey = key as keyof ParametersType;
        const defaultValue = DEFAULT_PARAMETERS[paramKey];
        const currentValue = parameters[paramKey];
        return currentValue === defaultValue;
    });
}

// --- Derived presets (partial overrides on top of DEFAULT_PARAMETERS) ---
// These explicitly set present_day/present_horizon, so they're stable for their
// intended purpose (example trajectories, not forecaster records).

const derivedPresets: Record<string, Partial<ParametersType>> = {
  'ai2027-handcrafted': {
    // Handcrafted trajectory similar to AI 2027
    present_doubling_time: 0.3301260512182007,
    ac_time_horizon_minutes: 6.324849176843143,
    doubling_difficulty_growth_factor: 0.85,
    ai_research_taste_slope: 2.7669416454115114,
    present_day: 2025.2,
    present_horizon: 15.311434662209903,
  },
};

// --- Centralized preset registry ---
// Combines frozen snapshots, derived presets, and backwards-compat aliases.
export const PRESET_BASELINES: Record<string, ParametersType> = {
  // Frozen forecaster snapshots (stable forever)
  ...FROZEN_PRESETS,
  // Derived presets (merged with DEFAULT_PARAMETERS)
  ...Object.fromEntries(
    Object.entries(derivedPresets).map(([id, overrides]) => [
      id,
      { ...DEFAULT_PARAMETERS, ...overrides } as ParametersType,
    ])
  ),
  // Backwards-compat aliases for old URLs
  'baseline': FROZEN_PRESETS['daniel-08-16-26'],
  // URL-compat aliases: share links minted while the 08-07-26 preset ids were
  // live must keep resolving; values were identical to the 08-10-26 presets.
  'daniel-08-07-26': FROZEN_PRESETS['daniel-08-16-26'],
  'eli-08-07-26': FROZEN_PRESETS['eli-08-16-26'],
  'daniel-08-10-26': FROZEN_PRESETS['daniel-08-16-26'],
  'eli-08-10-26': FROZEN_PRESETS['eli-08-16-26'],
  'brendan-08-10-26': FROZEN_PRESETS['brendan-08-16-26'],
  'eli-median': FROZEN_PRESETS['eli-12-29-25'],
};

// Dropdown items for preset selector UI
export const PARAMETER_PRESETS = [
  { id: 'daniel-08-16-26', label: "Daniel's median parameters (08/16/26)" },
  { id: 'eli-08-16-26', label: "Eli's median parameters (08/16/26)" },
  { id: 'brendan-08-16-26', label: "Brendan's median parameters (08/16/26)" },
  { id: 'daniel-04-02-26', label: "Daniel's median parameters (04/02/26)" },
  { id: 'daniel-12-29-25', label: "Daniel's median parameters (12/29/25)" },
  { id: 'eli-04-02-26', label: "Eli's median parameters (04/02/26)" },
  { id: 'eli-12-29-25', label: "Eli's median parameters (12/29/25)" },
  { id: 'ai2027-handcrafted', label: 'Example trajectory similar to AI 2027 (handcrafted)' },
];

// Baseline assumed for old URLs without a `base` param
export const LEGACY_BASELINE_ID = 'eli-12-29-25';

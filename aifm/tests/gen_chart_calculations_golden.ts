// Generates the cross-language golden fixture for the closed-form algebra in
// utils/chartCalculations.ts — the module that powers the slider-hover preview
// charts (components/SliderPopover.tsx). It never integrates the ODE; it
// evaluates algebra whose constants are, per that file's own header, "kept in
// step with Python progress_model/config.py by hand". This fixture is what
// detects the two languages drifting apart.
//
// Two families of cases are recorded, both by running the REAL exported
// frontend functions:
//
//   horizon_cases — computeHorizonFromProgress(progress, params, anchorProgress?)
//     Python counterpart: progress_model.ac_anchor._make_decaying_horizon_observable(
//         H_0, A_0, T_0, anchor_progress).predict(progress)
//     with H_0 = present_horizon, A_0 = 1 - doubling_difficulty_growth_factor,
//     T_0 = the doubling time.
//
//   sc_cases — generateSCHorizonData(params, config, startOOM, length)
//     Its automationFraction channel is AutomationModel.get_automation_fraction
//     (reached through progress_model.progress_rate.compute_automation_fraction)
//     on a linear schedule; its progressAtSc is the `invert` half of the same
//     decaying-doubling-time observable used above.
//
//   The aiResearchTaste channel of generateSCHorizonData is deliberately NOT
//   recorded: the TS TasteDistribution is a plain log-normal, while the Python
//   TasteDistribution applies a bounded transform and fits (mu, sigma) by
//   least squares. They are not counterparts — see the note in
//   tests/test_chart_calculations_cross_language.py.
//
// Every parameter combination here stays inside the domain where both
// implementations take their main analytic branch. The fallback branches are
// known to disagree (TS falls back to H_0, Python to 1e12) and are documented,
// not pinned.
//
// Regenerate (only with explicit permission, since it is golden data):
//   npx tsx tests/gen_chart_calculations_golden.ts
//
// Run from the repo root.

import { writeFileSync } from 'node:fs';
import {
  computeHorizonFromProgress,
  createDefaultSCHorizonConfig,
  generateSCHorizonData,
} from '../utils/chartCalculations.ts';
import { DEFAULT_PARAMETERS, type ParametersType } from '../constants/parameters.ts';

// Only the fields these cases vary. Spelled out rather than
// Partial<ParametersType> because ParametersType carries a string index
// signature, and Partial would widen it to include undefined.
interface ParamOverrides {
  present_horizon?: number;
  present_doubling_time?: number;
  doubling_difficulty_growth_factor?: number;
  ac_time_horizon_minutes?: number;
  automation_fraction_at_coding_automation_anchor?: number;
}

function withParams(overrides: ParamOverrides): ParametersType {
  return { ...DEFAULT_PARAMETERS, ...overrides };
}

// ---------------------------------------------------------------------------
// computeHorizonFromProgress
// ---------------------------------------------------------------------------

interface HorizonCase {
  name: string;
  present_horizon: number;
  present_doubling_time: number;
  doubling_difficulty_growth_factor: number;
  anchor_progress: number | null;
  progress: number[];
}

// A_0 = 1 - doubling_difficulty_growth_factor, so g = 0.92 -> A_0 = 0.08 (the
// shipping default), g = 1.0 would be A_0 = 0 (excluded: the two languages take
// different branches there), g > 1 gives a negative A_0 (accelerating doublings,
// which both sides explicitly allow).
const horizonCases: HorizonCase[] = [
  {
    // Shipping defaults, unshifted form.
    name: 'default_unshifted',
    present_horizon: 186.0,
    present_doubling_time: 0.37488146126475763,
    doubling_difficulty_growth_factor: 0.92,
    anchor_progress: null,
    // Spans well below 0 up to the point where the base term
    // (1 - A_0 * p / T_0) reaches zero at p = T_0 / A_0 = 4.686, and past it
    // where both sides clamp the base to 1e-12.
    progress: [-4, -1, -0.25, 0, 0.25, 1, 2, 3, 4, 4.6, 4.686, 4.6860182658094705, 5, 8],
  },
  {
    // Shifted form: the backend passes an anchor_progress and the curve is
    // evaluated on (p - anchor).
    name: 'default_shifted_anchor_3_1',
    present_horizon: 186.0,
    present_doubling_time: 0.37488146126475763,
    doubling_difficulty_growth_factor: 0.92,
    anchor_progress: 3.1,
    progress: [-2, 0, 3.0, 3.1, 3.2, 5, 7.7, 7.786018265809471, 8.2, 12],
  },
  {
    // Legacy preset values (smaller H_0, shorter doubling time).
    name: 'legacy_small_h0',
    present_horizon: 32.0,
    present_doubling_time: 0.458,
    doubling_difficulty_growth_factor: 0.92,
    anchor_progress: null,
    progress: [-3, -0.5, 0, 0.75, 2.5, 5, 5.724, 5.725, 9],
  },
  {
    // Negative A_0 (difficulty growth > 1): base term grows, exponent flips
    // sign. Both implementations allow A_0 < 0 explicitly.
    name: 'negative_a0',
    present_horizon: 69.9,
    present_doubling_time: 0.3328402169209725,
    doubling_difficulty_growth_factor: 1.05,
    anchor_progress: null,
    progress: [-5, -1, 0, 1, 2, 4, 8],
  },
  {
    // A_0 close to 1 (very fast-decaying doublings) — a large negative
    // exponent, so the tail values are enormous but finite on both sides.
    name: 'a0_near_one',
    present_horizon: 186.0,
    present_doubling_time: 1.0,
    doubling_difficulty_growth_factor: 0.05,
    anchor_progress: null,
    progress: [-1, 0, 0.2, 0.5, 0.9, 1.0, 1.0526315789473684, 1.2],
  },
  {
    // Tiny A_0 (nearly constant doubling time) — the branch that is closest to
    // the pure-exponential limit without hitting the excluded A_0 == 0 case.
    name: 'a0_tiny',
    present_horizon: 186.0,
    present_doubling_time: 0.4,
    doubling_difficulty_growth_factor: 0.999999,
    anchor_progress: null,
    progress: [-4, 0, 1, 4, 10, 40],
  },
  {
    // Long doubling time and a large shifted anchor, i.e. the regime the
    // /forecast sliders reach at the top of their range.
    name: 'long_doubling_shifted',
    present_horizon: 500.0,
    present_doubling_time: 2.5,
    doubling_difficulty_growth_factor: 0.85,
    anchor_progress: -1.75,
    progress: [-6, -1.75, 0, 4, 12, 14.916666666666666, 20],
  },
];

const horizon_cases = horizonCases.map((c) => {
  const params = withParams({
    present_horizon: c.present_horizon,
    present_doubling_time: c.present_doubling_time,
    doubling_difficulty_growth_factor: c.doubling_difficulty_growth_factor,
  });
  return {
    ...c,
    // A_0/T_0/H_0 spelled out so the Python side feeds the observable exactly
    // the numbers the TS side used, with no unit conversion in between.
    H_0: c.present_horizon,
    T_0: c.present_doubling_time,
    A_0: 1 - c.doubling_difficulty_growth_factor,
    horizon: c.progress.map((p) =>
      computeHorizonFromProgress(
        p,
        params,
        c.anchor_progress === null ? undefined : c.anchor_progress,
      ),
    ),
  };
});

// ---------------------------------------------------------------------------
// generateSCHorizonData: automation fraction + progress_at_aa inversion
// ---------------------------------------------------------------------------

interface SCCase {
  name: string;
  ac_time_horizon_minutes: number;
  automation_fraction_at_coding_automation_anchor: number;
  present_horizon: number;
  present_doubling_time: number;
  doubling_difficulty_growth_factor: number;
  startOOM: number;
  length: number;
  // SliderPopover overwrites config.anchorProgress when the backend reports a
  // shifted horizon fit; null leaves the unshifted form.
  anchorProgress?: number;
}

const scCases: SCCase[] = [
  // Shipping defaults. The OOM grid deliberately runs from far below the
  // reference OOMs (progress < 0, automation clips at 0) to far above the
  // upper anchor (automation clips at 1).
  {
    name: 'default',
    ac_time_horizon_minutes: 4.5,
    automation_fraction_at_coding_automation_anchor: 0.95,
    present_horizon: 186.0,
    present_doubling_time: 0.37488146126475763,
    doubling_difficulty_growth_factor: 0.92,
    startOOM: 22,
    length: 16,
  },
  // Short AC horizon -> small progress_at_aa -> steep automation ramp.
  {
    name: 'short_ac_horizon',
    ac_time_horizon_minutes: 3.0,
    automation_fraction_at_coding_automation_anchor: 0.8,
    present_horizon: 186.0,
    present_doubling_time: 0.37488146126475763,
    doubling_difficulty_growth_factor: 0.92,
    startOOM: 24,
    length: 12,
  },
  // Long AC horizon -> large progress_at_aa -> shallow automation ramp.
  {
    name: 'long_ac_horizon',
    ac_time_horizon_minutes: 6.5,
    automation_fraction_at_coding_automation_anchor: 1.0,
    present_horizon: 186.0,
    present_doubling_time: 0.37488146126475763,
    doubling_difficulty_growth_factor: 0.92,
    startOOM: 20,
    length: 20,
  },
  // The exact grid the live UI uses: SliderPopover calls
  // generateSCHorizonData(params, config, TRAINING_COMPUTE_REFERENCE_OOMS - 3)
  // with the default length of 20.
  {
    name: 'live_grid_unshifted',
    ac_time_horizon_minutes: 4.5,
    automation_fraction_at_coding_automation_anchor: 0.95,
    present_horizon: 186.0,
    present_doubling_time: 0.37488146126475763,
    doubling_difficulty_growth_factor: 0.92,
    startOOM: 23.54,
    length: 20,
  },
  // Same grid with the shifted horizon form, which is what SliderPopover
  // installs whenever the backend reports uses_shifted_form.
  {
    name: 'live_grid_shifted_anchor',
    ac_time_horizon_minutes: 4.5,
    automation_fraction_at_coding_automation_anchor: 0.95,
    present_horizon: 186.0,
    present_doubling_time: 0.37488146126475763,
    doubling_difficulty_growth_factor: 0.92,
    startOOM: 23.54,
    length: 20,
    anchorProgress: 3.1,
  },
  // Legacy horizon parameters with a smaller upper automation anchor, so the
  // linear schedule never reaches 1 inside the anchor interval.
  {
    name: 'legacy_params_low_upper_anchor',
    ac_time_horizon_minutes: 5.0,
    automation_fraction_at_coding_automation_anchor: 0.6,
    present_horizon: 32.0,
    present_doubling_time: 0.458,
    doubling_difficulty_growth_factor: 0.92,
    startOOM: 25,
    length: 10,
  },
];

const sc_cases = scCases.map((c) => {
  const params = withParams({
    ac_time_horizon_minutes: c.ac_time_horizon_minutes,
    automation_fraction_at_coding_automation_anchor:
      c.automation_fraction_at_coding_automation_anchor,
    present_horizon: c.present_horizon,
    present_doubling_time: c.present_doubling_time,
    doubling_difficulty_growth_factor: c.doubling_difficulty_growth_factor,
  });
  const config = createDefaultSCHorizonConfig(params);
  if (c.anchorProgress !== undefined) {
    config.anchorProgress = c.anchorProgress;
  }
  const out = generateSCHorizonData(params, config, c.startOOM, c.length);
  return {
    ...c,
    H_0: config.presentHorizon,
    T_0: config.presentDoublingTime,
    A_0: 1 - config.doublingDifficultyGrowthRate,
    anchor_progress: config.anchorProgress ?? null,
    ac_horizon_minutes: Math.pow(10, c.ac_time_horizon_minutes),
    // The lower automation anchor is hardcoded in createDefaultSCHorizonConfig;
    // record it so the Python AutomationModel is fitted on the same two points.
    lower_anchor_progress: config.automationAnchors.lowerProgress,
    lower_anchor_automation: config.automationAnchors.lowerAutomation,
    // The OOM -> progress offset. It has no live Python counterpart (the Python
    // constant is commented out in progress_model/config.py), so the Python
    // side consumes the progress values rather than re-deriving them.
    progress_offset: config.trainingComputeReferenceOOMs,
    effective_computes: out.effectiveComputes,
    progress: out.effectiveComputes.map((oom) => oom - config.trainingComputeReferenceOOMs),
    automation_fraction: out.automationFraction,
    progress_at_sc: out.progressAtSc,
  };
});

const outObj = {
  _comment:
    'Cross-language golden fixture for utils/chartCalculations.ts. Generated by ' +
    'tests/gen_chart_calculations_golden.ts running the real exported frontend ' +
    'functions. Consumed by tests/test_chart_calculations_cross_language.py. ' +
    'Do not hand-edit.',
  horizon_cases,
  sc_cases,
};

const path = new URL('./golden_data/chart_calculations_golden.json', import.meta.url);
writeFileSync(path, JSON.stringify(outObj, null, 2) + '\n');
console.log(
  `Wrote ${horizon_cases.length} horizon cases and ${sc_cases.length} SC cases to ${path.pathname}`,
);

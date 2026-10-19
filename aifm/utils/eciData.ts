// GENERATED FILE -- do not edit by hand.
// Regenerate with: python scripts/bridge/epoch_eci_to_ts.py
// Source of truth: epoch_capabilities_index.csv
//
// Data: Epoch AI, "AI Benchmarking Hub" (https://epoch.ai/benchmarks), CC-BY 4.0.
// Cite as: Epoch AI, 'AI Benchmarking Hub'. Published online at epoch.ai.
// CSV ships inside https://epoch.ai/data/benchmark_data.zip
//
// Reduced to the ECI FRONTIER: the running maximum of score over release date,
// one point per date, strictly increasing. All organizations compete for it.
// 21 frontier models out of 518 scored-and-dated rows (815 rows total).
//
// DISPLAY ONLY: overlaid on the Epoch Capabilities Index chart for comparison.
// Never fed to the model or to any fit.

export interface EciFrontierRow {
  /** ISO yyyy-mm-dd release date of the model. */
  releaseDate: string;
  /** Epoch Capabilities Index score (dimensionless, ~100-200 over the covered era). */
  eciScore: number;
  /** Lab that released the model, verbatim from Epoch. */
  organization: string;
  /** Human-readable model name for the tooltip. */
  displayName: string;
  /** Epoch's model version identifier. */
  modelVersion: string;
}

export const EPOCH_ECI_SOURCE_URL = "https://epoch.ai/benchmarks";
export const EPOCH_ECI_ATTRIBUTION = "Epoch AI, AI Benchmarking Hub (CC-BY 4.0)";
/** Newest frontier release in this snapshot; the CSV's vintage in one value. */
export const EPOCH_ECI_LATEST_RELEASE_DATE = "2026-07-09";

export const eciFrontierRows: EciFrontierRow[] = [
  {
    releaseDate: "2023-02-24",
    eciScore: 109.24,
    organization: "Meta AI",
    displayName: "LLaMA-65B",
    modelVersion: "LLaMA-65B",
  },
  {
    releaseDate: "2023-03-14",
    eciScore: 125.7,
    organization: "OpenAI",
    displayName: "GPT-4 (Mar 2023)",
    modelVersion: "gpt-4-32k-0314",
  },
  {
    releaseDate: "2023-11-06",
    eciScore: 125.83,
    organization: "OpenAI",
    displayName: "GPT-4 Turbo (Nov 2023)",
    modelVersion: "gpt-4-turbo",
  },
  {
    releaseDate: "2024-02-29",
    eciScore: 126.65,
    organization: "Anthropic",
    displayName: "Claude 3 Opus",
    modelVersion: "claude-3-opus-20240229",
  },
  {
    releaseDate: "2024-04-09",
    eciScore: 127.27,
    organization: "OpenAI",
    displayName: "GPT-4 Turbo (Apr 2024)",
    modelVersion: "gpt-4-turbo-2024-04-09",
  },
  {
    releaseDate: "2024-05-13",
    eciScore: 128.57,
    organization: "OpenAI",
    displayName: "GPT-4o (May 2024)",
    modelVersion: "gpt-4o-2024-05-13",
  },
  {
    releaseDate: "2024-06-20",
    eciScore: 130.0,
    organization: "Anthropic",
    displayName: "Claude 3.5 Sonnet (Jun 2024)",
    modelVersion: "claude-3-5-sonnet-20240620",
  },
  {
    releaseDate: "2024-09-12",
    eciScore: 136.45,
    organization: "OpenAI",
    displayName: "o1-mini (unknown thinking)",
    modelVersion: "o1-mini-2024-09-12_unknown",
  },
  {
    releaseDate: "2024-12-17",
    eciScore: 142.45,
    organization: "OpenAI",
    displayName: "o1 (unknown thinking)",
    modelVersion: "o1-2024-12-17_unknown",
  },
  {
    releaseDate: "2025-03-25",
    eciScore: 144.53,
    organization: "Google DeepMind",
    displayName: "Gemini 2.5 Pro Exp (Mar 2025)",
    modelVersion: "gemini-2.5-pro-exp-03-25",
  },
  {
    releaseDate: "2025-04-16",
    eciScore: 146.9,
    organization: "OpenAI",
    displayName: "o3 (unknown thinking)",
    modelVersion: "o3-2025-04-16_unknown",
  },
  {
    releaseDate: "2025-06-10",
    eciScore: 147.54,
    organization: "OpenAI",
    displayName: "o3-pro",
    modelVersion: "o3-pro-2025-06-10_medium",
  },
  {
    releaseDate: "2025-08-07",
    eciScore: 150.0,
    organization: "OpenAI",
    displayName: "GPT-5 (unknown thinking)",
    modelVersion: "gpt-5-2025-08-07_unknown",
  },
  {
    releaseDate: "2025-10-07",
    eciScore: 150.33,
    organization: "OpenAI",
    displayName: "GPT-5 Pro",
    modelVersion: "gpt-5-pro-2025-10-06_unknown",
  },
  {
    releaseDate: "2025-11-18",
    eciScore: 153.2,
    organization: "Google DeepMind",
    displayName: "Gemini 3 Pro Preview",
    modelVersion: "gemini-3-pro-preview",
  },
  {
    releaseDate: "2025-12-11",
    eciScore: 155.03,
    organization: "OpenAI",
    displayName: "GPT-5.2 Pro",
    modelVersion: "gpt-5.2-pro-2025-12-11_xhigh",
  },
  {
    releaseDate: "2026-02-05",
    eciScore: 155.69,
    organization: "OpenAI",
    displayName: "GPT-5.3 Codex (xhigh)",
    modelVersion: "gpt-5.3-codex_xhigh",
  },
  {
    releaseDate: "2026-03-05",
    eciScore: 158.46,
    organization: "OpenAI",
    displayName: "GPT-5.4 Pro (xhigh)",
    modelVersion: "gpt-5.4-pro-2026-03-05_xhigh",
  },
  {
    releaseDate: "2026-04-23",
    eciScore: 161.49,
    organization: "OpenAI",
    displayName: "GPT-5.5 Pro (xhigh)",
    modelVersion: "gpt-5.5-pro_xhigh",
  },
  {
    releaseDate: "2026-06-09",
    eciScore: 161.53,
    organization: "Anthropic",
    displayName: "Claude Fable 5 (xhigh)",
    modelVersion: "claude-fable-5_xhigh",
  },
  {
    releaseDate: "2026-07-09",
    eciScore: 161.65,
    organization: "OpenAI",
    displayName: "GPT-5.6 Sol (xhigh)",
    modelVersion: "gpt-5.6-sol_xhigh",
  },
];

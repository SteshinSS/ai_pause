// GENERATED FILE -- do not edit by hand.
// Regenerate with: python scripts/bridge/epoch_revenue_to_ts.py
// Source of truth: epoch_anthropic_revenue.csv
//
// Data: Epoch AI, "Data on AI Companies" (https://epoch.ai/data/ai-companies), CC-BY 4.0.
// Cite as: Josh You, John Croxton, Venkat Somala, Yafah Edelman,
// 'Data on AI Companies'. Published online at epoch.ai.
//
// DISPLAY ONLY: these are observed, reported revenue figures overlaid on the
// Predicted AI Revenue chart. They are never fed to the model or to any fit.

export interface ObservedRevenueRow {
  /** Company the figure refers to. */
  company: string;
  /** Date the figure describes (ISO yyyy-mm-dd), not the date it was reported. */
  date: string;
  /** Annualized revenue in US DOLLARS (not millions, not billions, not trillions). */
  annualizedRevenueUsd: number;
  /** e.g. 'Annualized run rate' or 'Annual recurring revenue (ARR)'. */
  revenueType: string;
  /** Epoch's confidence in the figure, e.g. 'Confident' / 'Likely'. */
  confidence: string;
  /** e.g. 'Company disclosure' / 'Media report'. */
  sourceType: string;
  sourceUrl: string;
  /** Date the figure was reported. */
  reportDate: string;
}

export const EPOCH_REVENUE_SOURCE_URL = "https://epoch.ai/data/ai-companies";
export const EPOCH_REVENUE_ATTRIBUTION = "Epoch AI, Data on AI Companies (CC-BY 4.0)";

export const observedRevenueRows: ObservedRevenueRow[] = [
  {
    company: "Anthropic",
    date: "2023-10-01",
    annualizedRevenueUsd: 100000000,
    revenueType: "Annualized run rate",
    confidence: "Likely",
    sourceType: "Media report",
    sourceUrl: "https://www.theinformation.com/articles/openai-rival-anthropic-in-talks-to-raise-2-billion-from-google-others-as-ai-arms-race-accelerates?rc=4ylmkj",
    reportDate: "2023-10-03",
  },
  {
    company: "Anthropic",
    date: "2024-01-01",
    annualizedRevenueUsd: 87000000,
    revenueType: "Annualized run rate",
    confidence: "Confident",
    sourceType: "Company disclosure",
    sourceUrl: "https://www.anthropic.com/news/anthropic-expands-global-leadership-in-enterprise-ai-naming-chris-ciauri-as-managing-director-of",
    reportDate: "2025-09-26",
  },
  {
    company: "Anthropic",
    date: "2024-12-31",
    annualizedRevenueUsd: 1000000000,
    revenueType: "Annualized run rate",
    confidence: "Likely",
    sourceType: "Media report",
    sourceUrl: "https://www.cnbc.com/2025/01/22/google-agrees-to-new-1-billion-investment-in-anthropic.html",
    reportDate: "2025-01-22",
  },
  {
    company: "Anthropic",
    date: "2025-03-01",
    annualizedRevenueUsd: 1400000000,
    revenueType: "Annualized run rate",
    confidence: "Likely",
    sourceType: "Media report",
    sourceUrl: "https://www.theinformation.com/articles/anthropics-claude-drives-strong-revenue-growth-while-powering-manus-sensation?rc=9mzoog",
    reportDate: "2025-03-11",
  },
  {
    company: "Anthropic",
    date: "2025-03-31",
    annualizedRevenueUsd: 2000000000,
    revenueType: "Annualized run rate",
    confidence: "Confident",
    sourceType: "Company disclosure",
    sourceUrl: "https://www.cnbc.com/2025/05/16/anthropic-ai-credit-facility.html",
    reportDate: "2025-05-16",
  },
  {
    company: "Anthropic",
    date: "2025-05-30",
    annualizedRevenueUsd: 3000000000,
    revenueType: "Annualized run rate",
    confidence: "Likely",
    sourceType: "Media report",
    sourceUrl: "https://www.cnbc.com/2025/05/30/anthropic-hits-3-billion-in-annualized-revenue-on-business-demand-for-ai.html",
    reportDate: "2025-05-30",
  },
  {
    company: "Anthropic",
    date: "2025-07-01",
    annualizedRevenueUsd: 4000000000,
    revenueType: "Annualized run rate",
    confidence: "Likely",
    sourceType: "Media report",
    sourceUrl: "https://www.theinformation.com/articles/anthropic-revenue-hits-4-billion-annual-pace-competition-cursor-intensifies?rc=9mzoog",
    reportDate: "2025-07-01",
  },
  {
    company: "Anthropic",
    date: "2025-07-29",
    annualizedRevenueUsd: 5000000000,
    revenueType: "Annual recurring revenue (ARR)",
    confidence: "Likely",
    sourceType: "Company disclosure",
    sourceUrl: "https://www.bloomberg.com/news/articles/2025-07-29/anthropic-nears-deal-to-raise-funding-at-170-billion-valuation",
    reportDate: "2025-07-29",
  },
  {
    company: "Anthropic",
    date: "2025-10-21",
    annualizedRevenueUsd: 7000000000,
    revenueType: "Annualized run rate",
    confidence: "Confident",
    sourceType: "Company disclosure",
    sourceUrl: "https://www.anthropic.com/news/statement-dario-amodei-american-ai-leadership",
    reportDate: "2025-10-21",
  },
  {
    company: "Anthropic",
    date: "2025-12-31",
    annualizedRevenueUsd: 9000000000,
    revenueType: "Annualized run rate",
    confidence: "Likely",
    sourceType: "Media report",
    sourceUrl: "https://www.bloomberg.com/news/articles/2026-01-21/anthropic-s-revenue-run-rate-tops-9-billion-as-vcs-pile-in",
    reportDate: "2026-01-21",
  },
  {
    company: "Anthropic",
    date: "2026-02-12",
    annualizedRevenueUsd: 14000000000,
    revenueType: "Annualized run rate",
    confidence: "Confident",
    sourceType: "Company disclosure",
    sourceUrl: "https://www.anthropic.com/news/anthropic-raises-30-billion-series-g-funding-380-billion-post-money-valuation",
    reportDate: "2026-02-12",
  },
  {
    company: "Anthropic",
    date: "2026-03-03",
    annualizedRevenueUsd: 19000000000,
    revenueType: "Annualized run rate",
    confidence: "Confident",
    sourceType: "Company disclosure,Media report",
    sourceUrl: "https://x.com/apoorv03/status/2029018856342290550?s=20",
    reportDate: "2026-03-03",
  },
  {
    company: "Anthropic",
    date: "2026-04-06",
    annualizedRevenueUsd: 30000000000,
    revenueType: "Annualized run rate",
    confidence: "Confident",
    sourceType: "Company disclosure",
    sourceUrl: "https://www.anthropic.com/news/google-broadcom-partnership-compute",
    reportDate: "2026-04-06",
  },
  {
    company: "Anthropic",
    date: "2026-05-15",
    annualizedRevenueUsd: 47000000000,
    revenueType: "Annualized run rate",
    confidence: "Likely",
    sourceType: "Company disclosure",
    sourceUrl: "https://www.anthropic.com/news/series-h",
    reportDate: "2026-05-28",
  },
  {
    company: "Anthropic",
    date: "2026-07-31",
    annualizedRevenueUsd: 65000000000,
    revenueType: "Annualized run rate",
    confidence: "Likely",
    sourceType: "Media report",
    sourceUrl: "https://www.bloomberg.com/news/articles/2026-08-17/anthropic-revenue-run-rate-surpasses-65-billion-ahead-of-ipo",
    reportDate: "2026-08-17",
  },
];

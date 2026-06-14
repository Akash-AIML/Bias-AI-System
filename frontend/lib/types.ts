export type GroupMetric = {
  accuracy: number;
  selection_rate: number;
  fpr?: number;   // False Positive Rate per group
  fnr?: number;   // False Negative Rate per group
};

export type AuditReport = {
  mode: string;
  verdict: string;
  metrics: {
    dp_diff: number;
    eo_diff: number;
    bsi_score?: number;
    disparate_impact_ratio?: number;  // EEOC 4/5ths rule (< 0.8 = adverse impact)
  };
  group_metrics: Record<string, GroupMetric>;
  warnings: string[];
  target_transformation: TargetTransformation;
  reliability_note: string;
  risk_tier?: RiskTier;
  proxy_features?: ProxyFeature[];
  temporal_drift?: TemporalDrift;
  text_bias?: TextBias;
  legal_context?: LegalContext;
  audit_narrative?: AuditNarrative;
  mitigation_simulation?: MitigationSimulation[];
};

export type MitigationSimulation = {
  name: string;
  bsi_score: number;
  dp_diff: number;
  eo_diff: number;
};

export type TargetTransformation = {
  source_type: string;
  method: string;
  mapping: Record<string, number>;
  threshold: number | null;
};

export type RiskTier = {
  level: string;
  label: string;
  action: string;
};

export type ProxyFeature = {
  feature: string;
  correlation: number;
  direction: string;
};

export type TemporalDrift = {
  status: string;
  time_column: string | null;
  early_bsi: number | null;
  late_bsi: number | null;
  delta: number | null;
};

export type TextSentimentGap = {
  column: string;
  group_a: string;
  group_b: string;
  gap: number;
};

export type TextTopTerms = {
  column: string;
  group: string;
  terms: string[];
};

export type TextBias = {
  has_text_columns: boolean;
  columns: string[];
  sentiment_gaps: TextSentimentGap[];
  top_terms: TextTopTerms[];
  summary: string;
};

export type LegalContext = {
  domain: string;
  framework: string;
  notes: string;
};

export type AuditNarrative = {
  executive_summary: string;
  findings: string[];
  risk_assessment: string;
  legal_exposure: string;
  recommended_actions: string[];
  counterfactuals: string[];
};

// ── New enriched LLM output types ──

/** Taxonomy-validated bias label returned by the AI. */
export type BiasType =
  | 'Selection Bias'
  | 'Historical Bias'
  | 'Proxy Discrimination'
  | 'Measurement Bias'
  | 'Label Bias'
  | 'Representation Bias'
  | 'Algorithmic Amplification'
  | 'Temporal Drift Bias'
  | 'Text/Language Bias';

/** AI-generated per-proxy-feature analysis with remediation. */
export type ProxyAnalysisItem = {
  feature: string;
  correlation: number;
  direction: 'positive' | 'negative';
  strength: 'strong' | 'moderate' | 'weak';
  why_it_matters: string;
  how_to_fix: string;
};

/** BSI band, plain-English meaning, and drift trend from AI. */
export type BSIInterpretation = {
  score: number;
  band: string;        // e.g. "HIGH (40-65): significant bias present"
  meaning: string;     // plain-English sentence for this specific dataset
  trend: string;       // e.g. "worsening", "stable", "not_available"
};

/** One step in the AI-generated 4-step prioritised rectification plan. */
export type RectificationStep = {
  priority: number;          // 1 = highest
  action: string;            // short title
  description: string;       // 1-2 sentences citing specific features/groups
  expected_impact: string;   // e.g. "Reduces DP gap by ~8%"
  timeline: string;          // e.g. "1-3 months"
};

export type AnalyzeResponse = {
  bias: boolean;
  audit_id: string;
  dp_diff: number;
  eo_diff: number;
  bsi_score: number;
  disparate_impact_ratio: number;  // EEOC 4/5ths rule: < 0.8 indicates adverse impact
  risk_tier: RiskTier;
  proxy_features: ProxyFeature[];
  temporal_drift: TemporalDrift;
  text_bias: TextBias;
  legal_context: LegalContext;
  audit_narrative: AuditNarrative;
  group_metrics: Record<string, GroupMetric>;
  suggestions: string[];
  explanation: string;
  summary: string;
  report_text: string;
  intent: string;
  domain: string;
  audit_mode: string;
  warnings: string[];
  info_notes: string[];
  weight_column: string | null;
  target_transformation: TargetTransformation;
  audit_report: AuditReport;
  mitigation_simulations: MitigationSimulation[];
  // ── enriched LLM output ──
  bias_types: BiasType[];
  proxy_analysis: ProxyAnalysisItem[];
  bsi_interpretation: BSIInterpretation;
  rectification_plan: RectificationStep[];
};

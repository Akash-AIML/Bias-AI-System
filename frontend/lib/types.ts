export type GroupMetric = {
  accuracy: number;
  selection_rate: number;
};

export type AuditReport = {
  mode: string;
  verdict: string;
  metrics: {
    dp_diff: number;
    eo_diff: number;
    bsi_score?: number;
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

export type AnalyzeResponse = {
  bias: boolean;
  audit_id: string;
  dp_diff: number;
  eo_diff: number;
  bsi_score: number;
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
};

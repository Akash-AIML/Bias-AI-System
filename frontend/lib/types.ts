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
  };
  group_metrics: Record<string, GroupMetric>;
  warnings: string[];
  reliability_note: string;
};

export type AnalyzeResponse = {
  bias: boolean;
  dp_diff: number;
  eo_diff: number;
  group_metrics: Record<string, GroupMetric>;
  suggestions: string[];
  explanation: string;
  summary: string;
  report_text: string;
  intent: string;
  domain: string;
  audit_mode: string;
  warnings: string[];
  audit_report: AuditReport;
};

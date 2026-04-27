export type GroupMetric = {
  accuracy: number;
  selection_rate: number;
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
};

"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { PanelRightOpen, Sparkles } from "lucide-react";
import { GeminiAgentDrawer } from "@/components/GeminiAgentDrawer";
import type { AgentMessage } from "@/components/GeminiAgentDrawer";

import { BiasChart } from "@/components/BiasChart";
import { GaugeChart } from "@/components/GaugeChart";
import { GroupMetricsChart } from "@/components/GroupMetricsChart";
import { MetricsCard } from "@/components/MetricsCard";
import { SimulationPanel } from "../../components/SimulationPanel";
import { SuggestionBox } from "@/components/SuggestionBox";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog } from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ThemeToggle } from "@/components/ThemeToggle";
import { getAnalysisFile, clearAnalysisFile } from "@/lib/analysis-cache";
import type { AnalyzeResponse } from "@/lib/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api";

function Typewriter({ text, speed = 8 }: { text: string; speed?: number }) {
  const [displayedText, setDisplayedText] = useState("");
  const [isDone, setIsDone] = useState(false);

  useEffect(() => {
    let index = 0;
    setDisplayedText("");
    setIsDone(false);

    if (!text) return;

    const interval = setInterval(() => {
      setDisplayedText(text.slice(0, index + 1));
      index++;
      if (index >= text.length) {
        clearInterval(interval);
        setIsDone(true);
      }
    }, speed);

    return () => clearInterval(interval);
  }, [text, speed]);

  return (
    <span className="leading-relaxed">
      {displayedText}
      {!isDone && (
        <span className="inline-block w-[2px] h-[1.1em] ml-1 align-middle bg-[var(--accent)] animate-pulse" />
      )}
    </span>
  );
}

export default function ResultsPage() {
  const router = useRouter();
  const [data, setData] = useState<AnalyzeResponse | null>(null);
  const [showDialog, setShowDialog] = useState(false);
  const [showSheet, setShowSheet] = useState(false);
  const [analysisMeta, setAnalysisMeta] = useState<Record<string, string> | null>(null);
  const [analysisFile, setAnalysisFile] = useState<File | null>(null);

  // Gemini Chat Agent States
  const [isAgentOpen, setIsAgentOpen] = useState(false);
  const [agentMessages, setAgentMessages] = useState<AgentMessage[]>([]);
  const [isAgentLoading, setIsAgentLoading] = useState(false);

  async function handleSendAgentMessage(messageText: string) {
    if (!data) return;

    const userMsg: AgentMessage = { role: "user", content: messageText };
    const updatedMessages = [...agentMessages, userMsg];
    setAgentMessages(updatedMessages);
    setIsAgentLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/agent`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: messageText,
          history: agentMessages,
          audit_report: data.audit_report,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to query Gemini Agent");
      }

      const payload = await response.json();
      setAgentMessages([
        ...updatedMessages,
        { role: "assistant", content: payload.response },
      ]);
    } catch (err) {
      setAgentMessages([
        ...updatedMessages,
        {
          role: "assistant",
          content: "Sorry, I had trouble reaching the auditing backend. Please verify that the server is running.",
        },
      ]);
    } finally {
      setIsAgentLoading(false);
    }
  }

  function handleClearAgentHistory() {
    setAgentMessages([]);
  }

  async function handleAskGeminiSuggestion(suggestionText: string) {
    setIsAgentOpen(true);
    const promptText = `Please provide a detailed, step-by-step implementation guide to address this mitigation suggestion:\n\n"${suggestionText}"`;
    await handleSendAgentMessage(promptText);
  }

  useEffect(() => {
    const raw = localStorage.getItem("analysisResult");
    if (!raw) {
      return;
    }

    try {
      const parsed = JSON.parse(raw) as AnalyzeResponse;
      setData(parsed);

      const metaRaw = localStorage.getItem("analysisMeta");
      if (metaRaw) {
        setAnalysisMeta(JSON.parse(metaRaw));
      }
      setAnalysisFile(getAnalysisFile());
    } catch {
      setData(null);
    }
  }, []);

  const groupRows = useMemo(() => {
    if (!data) {
      return [];
    }
    return Object.entries(data.group_metrics);
  }, [data]);

  const fairnessScore = useMemo(() => {
    if (!data) {
      return 0;
    }
    return Math.round(data.bsi_score);
  }, [data]);

  const riskLabel = useMemo(() => {
    return data?.risk_tier.label ?? "Risk";
  }, [data]);

  const riskVariant = useMemo(() => {
    if (!data) {
      return "secondary";
    }
    if (data.risk_tier.level === "green") {
      return "success";
    }
    if (data.risk_tier.level === "yellow") {
      return "secondary";
    }
    return "destructive";
  }, [data]);

  const auditModeLabel = useMemo(() => {
    if (!data) {
      return "Audit";
    }
    return data.audit_mode === "predictions" ? "Prediction audit" : "Proxy audit";
  }, [data]);

  const targetMappingRows = useMemo(() => {
    if (!data) {
      return [];
    }
    return Object.entries(data.target_transformation.mapping ?? {});
  }, [data]);

  if (!data) {
    return (
      <main>
        <div className="mx-auto max-w-4xl px-6 py-8">
          <h1 className="text-2xl font-bold">No Analysis Result Found</h1>
          <p className="text-slate-600">Run an analysis first from the upload page.</p>
          <Link href="/" className="mt-4 inline-block text-sm text-slate-700 underline">
            Back to Upload
          </Link>
        </div>
      </main>
    );
  }

  async function downloadReweightedCsv() {
    if (!analysisFile || !analysisMeta) {
      return;
    }

    const payload = new FormData();
    payload.append("file", analysisFile, analysisMeta.filename ?? analysisFile.name ?? "dataset.csv");
    payload.append("target", analysisMeta.target ?? "");
    payload.append("sensitive", analysisMeta.sensitive ?? "");
    if (analysisMeta.targetBinarizationThreshold?.trim()) {
      payload.append("target_binarization_threshold", analysisMeta.targetBinarizationThreshold.trim());
    }

    const response = await fetch(`${API_BASE_URL}/reweighted-csv`, {
      method: "POST",
      body: payload,
    });

    if (!response.ok) {
      return;
    }

    const downloadBlob = await response.blob();
    const url = window.URL.createObjectURL(downloadBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "reweighted_dataset.csv";
    link.click();
    window.URL.revokeObjectURL(url);
  }

  async function downloadReport() {
    if (!data) {
      return;
    }

    const payload = {
      dataset_summary: {
        file: analysisMeta?.filename ?? "-",
        target: analysisMeta?.target ?? "-",
        sensitive: analysisMeta?.sensitive ?? "-",
        prediction_column: analysisMeta?.predictionColumn ?? "-",
        org_name: analysisMeta?.orgName ?? "-",
        dataset_name: analysisMeta?.datasetName ?? "-",
        uploaded_at: analysisMeta?.uploadedAt ?? "-",
      },
      accountability_summary: {
        audit_id: data.audit_id,
      },
      verdict: data.bias ? "bias_detected" : "no_severe_bias",
      dp_diff: data.dp_diff,
      eo_diff: data.eo_diff,
      bsi_score: data.bsi_score,
      risk_tier: data.risk_tier,
      proxy_features: data.proxy_features,
      temporal_drift: data.temporal_drift,
      text_bias: data.text_bias,
      legal_context: data.legal_context,
      audit_narrative: data.audit_narrative,
      group_metrics: data.group_metrics,
      suggestions: data.suggestions,
      audit_mode: data.audit_mode,
    };

    const response = await fetch(`${API_BASE_URL}/report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      return;
    }

    const downloadBlob = await response.blob();
    const url = window.URL.createObjectURL(downloadBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "accountability_certificate.pdf";
    link.click();
    window.URL.revokeObjectURL(url);
  }

  return (
    <main>
      <div className="mx-auto max-w-5xl px-4 py-6 sm:px-6 lg:px-8 grid gap-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-4xl font-semibold tracking-tight text-[var(--text)] animate-fade-rise">
              Audit Dashboard
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-[var(--text-soft)] animate-fade-rise">
              Review bias severity, proxy discrimination signals, and compliance-ready findings.
            </p>
          </div>
        </div>

        <Card className="animate-fade-rise">
          <CardContent className="space-y-5 pt-6">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant={riskVariant}>{riskLabel}</Badge>
                <Badge variant="secondary">Bias Severity Index {fairnessScore}/100</Badge>
                <Badge variant="secondary">Audit ID {data.audit_id}</Badge>
                <Badge variant="secondary">{auditModeLabel}</Badge>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <Button onClick={() => setShowDialog(true)} size="lg">
                  Explain
                </Button>
                <Button variant="outline" onClick={downloadReweightedCsv} size="lg" disabled={!analysisFile}>
                  Download reweighted CSV
                </Button>
                <Button variant="outline" onClick={downloadReport} size="lg">
                  Download certificate
                </Button>
                <Button variant="outline" onClick={() => setShowSheet(true)} size="lg">
                  <PanelRightOpen size={16} className="mr-1" />
                  Dataset Details
                </Button>
                <Link href="/" className="text-sm text-[var(--text-soft)] underline">
                  Run another analysis
                </Link>
                <Button variant="ghost" onClick={() => {
                  localStorage.removeItem("analysisResult");
                  localStorage.removeItem("analysisMeta");
                  clearAnalysisFile();
                  router.push("/");
                }} size="sm">
                  Clear Analysis
                </Button>
              </div>
            </div>

            <Progress value={fairnessScore} />
          </CardContent>
        </Card>

        {data.warnings?.length ? (
          <Alert variant="destructive">
            <AlertTitle>Audit warnings</AlertTitle>
            <AlertDescription>
              <ul className="mt-2 list-disc pl-4">
                {data.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </AlertDescription>
          </Alert>
        ) : null}

        <Card>
          <CardHeader>
            <CardTitle>Target Transformation</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-[var(--text)]">
            <p>
              <span className="font-medium">Source type:</span> {data.target_transformation.source_type}
            </p>
            <p>
              <span className="font-medium">Method:</span> {data.target_transformation.method}
            </p>
            <p>
              <span className="font-medium">Threshold:</span>{" "}
              {data.target_transformation.threshold === null ? "Not used" : data.target_transformation.threshold}
            </p>
            <div>
              <p className="font-medium">Mapping</p>
              <ul className="mt-1 list-disc pl-5">
                {targetMappingRows.map(([label, value]) => (
                  <li key={label}>
                    {label} {"->"} {value}
                  </li>
                ))}
              </ul>
            </div>
          </CardContent>
        </Card>

        <Tabs defaultValue="metrics" className="animate-fade-rise">
          <TabsList>
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
            <TabsTrigger value="mitigation">Mitigation</TabsTrigger>
            <TabsTrigger value="accountability">Accountability</TabsTrigger>
            <TabsTrigger value="bias-tracer">Bias Tracer</TabsTrigger>
            <TabsTrigger value="temporal">Temporal Drift</TabsTrigger>
            <TabsTrigger value="language">Language Bias</TabsTrigger>
            <TabsTrigger value="explanation">Explanation</TabsTrigger>
          </TabsList>

          <TabsContent value="metrics" className="space-y-4">
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
              <MetricsCard label="Demographic Parity Diff" value={data.dp_diff.toFixed(4)} />
              <MetricsCard label="Equalized Odds Diff" value={data.eo_diff.toFixed(4)} />
              <MetricsCard label="BSI Score" value={data.bsi_score.toFixed(2)} helper={data.risk_tier.action} />
              <MetricsCard label="Intent" value={data.intent} helper={`Domain: ${data.domain}`} />
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <GaugeChart label="Demographic Parity" value={data.dp_diff} />
              <GaugeChart label="Equalized Odds" value={data.eo_diff} />
            </div>

            <BiasChart dpDiff={data.dp_diff} eoDiff={data.eo_diff} />
            <GroupMetricsChart metrics={data.group_metrics} />
            <SimulationPanel simulations={data.mitigation_simulations ?? []} />

            <Card>
              <CardHeader>
                <CardTitle>Group Metrics Table</CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="max-h-64">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Group</TableHead>
                        <TableHead>Accuracy</TableHead>
                        <TableHead>Selection Rate</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {groupRows.map(([groupName, metrics]) => (
                        <TableRow key={groupName}>
                          <TableCell>{groupName}</TableCell>
                          <TableCell>{metrics.accuracy.toFixed(4)}</TableCell>
                          <TableCell>{metrics.selection_rate.toFixed(4)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="mitigation">
            <SuggestionBox suggestions={data.suggestions} onAskGemini={handleAskGeminiSuggestion} />
          </TabsContent>

          <TabsContent value="accountability" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Legal Context</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-[var(--text)]">
                <p>
                  <span className="font-medium">Domain:</span> {data.legal_context.domain}
                </p>
                <p>
                  <span className="font-medium">Framework:</span> {data.legal_context.framework}
                </p>
                <p>{data.legal_context.notes}</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Audit Narrative</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-[var(--text)]">
                <p>{data.audit_narrative.executive_summary}</p>
                <Separator />
                <p>{data.audit_narrative.risk_assessment}</p>
                <p>{data.audit_narrative.legal_exposure}</p>
                <Separator />
                <div>
                  <p className="font-medium">Key Findings</p>
                  <ul className="mt-1 list-disc pl-5">
                    {data.audit_narrative.findings.map((finding) => (
                      <li key={finding}>{finding}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <p className="font-medium">Recommended Actions</p>
                  <ul className="mt-1 list-disc pl-5">
                    {data.audit_narrative.recommended_actions.map((action) => (
                      <li key={action}>{action}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <p className="font-medium">Counterfactuals</p>
                  <ul className="mt-1 list-disc pl-5">
                    {data.audit_narrative.counterfactuals.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="bias-tracer">
            <Card>
              <CardHeader>
                <CardTitle>Proxy Feature Correlations</CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Feature</TableHead>
                      <TableHead>Correlation</TableHead>
                      <TableHead>Direction</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.proxy_features.map((proxy) => (
                      <TableRow key={proxy.feature}>
                        <TableCell>{proxy.feature}</TableCell>
                        <TableCell>{proxy.correlation.toFixed(4)}</TableCell>
                        <TableCell>{proxy.direction}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="temporal">
            <Card>
              <CardHeader>
                <CardTitle>Temporal Drift</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-[var(--text)]">
                <p>
                  <span className="font-medium">Status:</span> {data.temporal_drift.status}
                </p>
                <p>
                  <span className="font-medium">Time Column:</span> {data.temporal_drift.time_column ?? "Not detected"}
                </p>
                <p>
                  <span className="font-medium">Early BSI:</span> {data.temporal_drift.early_bsi ?? "-"}
                </p>
                <p>
                  <span className="font-medium">Late BSI:</span> {data.temporal_drift.late_bsi ?? "-"}
                </p>
                <p>
                  <span className="font-medium">Delta:</span> {data.temporal_drift.delta ?? "-"}
                </p>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="language" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Language Bias Findings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-[var(--text)]">
                <p>{data.text_bias.summary}</p>
                <p>
                  <span className="font-medium">Text Columns:</span>{" "}
                  {data.text_bias.columns.length ? data.text_bias.columns.join(", ") : "None"}
                </p>
                <Separator />
                <div>
                  <p className="font-medium">Sentiment Gaps</p>
                  <ul className="mt-1 list-disc pl-5">
                    {data.text_bias.sentiment_gaps.map((gap) => (
                      <li key={`${gap.column}-${gap.group_a}-${gap.group_b}`}>
                        {gap.column}: {gap.group_a} vs {gap.group_b} gap {gap.gap.toFixed(3)}
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <p className="font-medium">Top Terms by Group</p>
                  <ul className="mt-1 list-disc pl-5">
                    {data.text_bias.top_terms.map((entry) => (
                      <li key={`${entry.column}-${entry.group}`}>
                        {entry.column} ({entry.group}): {entry.terms.join(", ") || "-"}
                      </li>
                    ))}
                  </ul>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="explanation">
            <Card>
              <CardHeader>
                <CardTitle>Explanation Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-[var(--text)]">
                <p className="text-base font-medium text-[var(--text)] leading-relaxed">
                  <Typewriter text={data.summary} />
                </p>
                <Separator />
                <p>{data.explanation}</p>
                <Separator />
                <p>{data.report_text}</p>
                <Separator />
                <p className="text-xs text-[var(--text-soft)]">
                  Audit mode: {auditModeLabel}. Target transformation: {data.target_transformation.method}.
                </p>
                <p className="text-xs text-[var(--text-soft)]">{data.audit_report.reliability_note}</p>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        <Dialog open={showDialog} onOpenChange={setShowDialog} title="LLM Explanation">
          <div className="grid gap-3">
            <div>
              <h4 className="mb-1 mt-0 text-sm font-semibold">Summary</h4>
              <p className="m-0 text-sm leading-6">{data.summary}</p>
            </div>
            <div>
              <h4 className="mb-1 mt-0 text-sm font-semibold">Explanation</h4>
              <p className="m-0 text-sm leading-6">{data.explanation}</p>
            </div>
            <div>
              <h4 className="mb-1 mt-0 text-sm font-semibold">Report</h4>
              <p className="m-0 text-sm leading-6">{data.report_text}</p>
            </div>
          </div>
        </Dialog>

        <Sheet open={showSheet} onOpenChange={setShowSheet}>
          <SheetContent>
            <SheetHeader>
              <SheetTitle>Dataset Summary</SheetTitle>
              <SheetDescription>Snapshot of the latest uploaded analysis request.</SheetDescription>
            </SheetHeader>

            <div className="space-y-3 text-sm text-slate-700">
              <div>
                <span className="font-medium">File:</span> {analysisMeta?.filename ?? "Unknown"}
              </div>
              <div>
                <span className="font-medium">Target:</span> {analysisMeta?.target ?? "Unknown"}
              </div>
              <div>
                <span className="font-medium">Sensitive:</span> {analysisMeta?.sensitive ?? "Unknown"}
              </div>
              <div>
                <span className="font-medium">Predictions:</span> {analysisMeta?.predictionColumn ?? "Proxy audit"}
              </div>
              <div>
                <span className="font-medium">Organization:</span> {analysisMeta?.orgName ?? "Not set"}
              </div>
              <div>
                <span className="font-medium">Dataset:</span> {analysisMeta?.datasetName ?? "Not set"}
              </div>
              <div>
                <span className="font-medium">Time column:</span> {analysisMeta?.timeColumn ?? "Auto"}
              </div>
              <div>
                <span className="font-medium">Text columns:</span> {analysisMeta?.textColumns ?? "Auto"}
              </div>
              <div>
                <span className="font-medium">Target threshold:</span>{" "}
                {analysisMeta?.targetBinarizationThreshold || "Median (auto)"}
              </div>
              <div>
                <span className="font-medium">Audit ID:</span> {data.audit_id}
              </div>
              <div>
                <span className="font-medium">Uploaded:</span> {analysisMeta?.uploadedAt ?? "-"}
              </div>
              <div>
                <span className="font-medium">Query:</span> {analysisMeta?.query ?? "check bias"}
              </div>
            </div>
          </SheetContent>
        </Sheet>
      </div>

      {/* Floating Action Button for Gemini Chat Agent */}
      <div className="fixed bottom-6 right-6 z-40 flex flex-col items-end gap-2 animate-fade-rise">
        <button
          onClick={() => setIsAgentOpen(true)}
          className="flex items-center gap-2 px-5 py-3.5 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white rounded-full shadow-lg hover:shadow-indigo-500/30 font-semibold transition-all transform hover:-translate-y-1 hover:scale-105 active:scale-95 duration-200 group border border-indigo-400/20"
        >
          <Sparkles size={16} className="text-indigo-200 group-hover:rotate-12 transition-transform" />
          <span>Ask Gemini Agent</span>
        </button>
      </div>

      {/* Slide-out Gemini Chat Drawer */}
      <GeminiAgentDrawer
        open={isAgentOpen}
        onOpenChange={setIsAgentOpen}
        auditReport={data.audit_report}
        messages={agentMessages}
        isLoading={isAgentLoading}
        onSubmitMessage={handleSendAgentMessage}
        onClearHistory={handleClearAgentHistory}
      />
    </main>
  );
}

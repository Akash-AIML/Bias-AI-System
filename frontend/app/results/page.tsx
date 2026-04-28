"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { PanelRightOpen } from "lucide-react";

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
import type { AnalyzeResponse } from "@/lib/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function ResultsPage() {
  const [data, setData] = useState<AnalyzeResponse | null>(null);
  const [showDialog, setShowDialog] = useState(false);
  const [showSheet, setShowSheet] = useState(false);
  const [analysisMeta, setAnalysisMeta] = useState<Record<string, string> | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);

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
      const fileRaw = localStorage.getItem("analysisFileContent");
      if (fileRaw) {
        setFileContent(fileRaw);
      }
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
    const penalty = Math.min(1, (Math.abs(data.dp_diff) + Math.abs(data.eo_diff)) / 2);
    return Math.round((1 - penalty) * 100);
  }, [data]);

  const riskLabel = useMemo(() => {
    if (fairnessScore >= 80) {
      return "Low risk";
    }
    if (fairnessScore >= 60) {
      return "Needs review";
    }
    return "Bias detected";
  }, [fairnessScore]);

  const auditModeLabel = useMemo(() => {
    if (!data) {
      return "Audit";
    }
    return data.audit_mode === "predictions" ? "Prediction audit" : "Proxy audit";
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
    if (!fileContent || !analysisMeta) {
      return;
    }

    const blob = new Blob([fileContent], { type: "text/csv" });
    const file = new File([blob], analysisMeta.filename ?? "dataset.csv", { type: "text/csv" });

    const payload = new FormData();
    payload.append("file", file);
    payload.append("target", analysisMeta.target ?? "");
    payload.append("sensitive", analysisMeta.sensitive ?? "");

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
      },
      verdict: data.bias ? "bias_detected" : "no_severe_bias",
      dp_diff: data.dp_diff,
      eo_diff: data.eo_diff,
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
    link.download = "fairness_report.pdf";
    link.click();
    window.URL.revokeObjectURL(url);
  }

  return (
    <main>
      <div className="mx-auto max-w-5xl px-4 py-6 sm:px-6 lg:px-8 grid gap-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-4xl font-semibold tracking-tight text-[var(--text)] animate-fade-rise">
              Bias Analysis Dashboard
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-[var(--text-soft)] animate-fade-rise">
              Review fairness metrics, mitigation guidance, and LLM interpretation.
            </p>
          </div>
          <ThemeToggle />
        </div>

        <Card className="animate-fade-rise">
          <CardContent className="space-y-5 pt-6">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant={data.bias ? "destructive" : "success"}>{riskLabel}</Badge>
                <Badge variant="secondary">Fairness score {fairnessScore}/100</Badge>
                <Badge variant="secondary">{auditModeLabel}</Badge>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <Button onClick={() => setShowDialog(true)} size="lg">
                  Explain
                </Button>
                <Button variant="outline" onClick={downloadReweightedCsv} size="lg">
                  Download reweighted CSV
                </Button>
                <Button variant="outline" onClick={downloadReport} size="lg">
                  Download PDF report
                </Button>
                <Button variant="outline" onClick={() => setShowSheet(true)} size="lg">
                  <PanelRightOpen size={16} className="mr-1" />
                  Dataset Details
                </Button>
                <Link href="/" className="text-sm text-[var(--text-soft)] underline">
                  Run another analysis
                </Link>
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

        <Tabs defaultValue="metrics" className="animate-fade-rise">
          <TabsList>
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
            <TabsTrigger value="mitigation">Mitigation</TabsTrigger>
            <TabsTrigger value="explanation">Explanation</TabsTrigger>
          </TabsList>

          <TabsContent value="metrics" className="space-y-4">
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
              <MetricsCard label="Demographic Parity Diff" value={data.dp_diff.toFixed(4)} />
              <MetricsCard label="Equalized Odds Diff" value={data.eo_diff.toFixed(4)} />
              <MetricsCard label="Intent" value={data.intent} helper={`Domain: ${data.domain}`} />
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <GaugeChart label="Demographic Parity" value={data.dp_diff} />
              <GaugeChart label="Equalized Odds" value={data.eo_diff} />
            </div>

            <BiasChart dpDiff={data.dp_diff} eoDiff={data.eo_diff} />
            <GroupMetricsChart metrics={data.group_metrics} />
            <SimulationPanel metrics={data.group_metrics} />

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
            <SuggestionBox suggestions={data.suggestions} />
          </TabsContent>

          <TabsContent value="explanation">
            <Card>
              <CardHeader>
                <CardTitle>Explanation Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-[var(--text)]">
                <p className="typewriter">{data.summary}</p>
                <Separator />
                <p>{data.explanation}</p>
                <Separator />
                <p>{data.report_text}</p>
                <Separator />
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
                <span className="font-medium">Query:</span> {analysisMeta?.query ?? "check bias"}
              </div>
            </div>
          </SheetContent>
        </Sheet>
      </div>
    </main>
  );
}

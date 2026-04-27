"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

import { BiasChart } from "@/components/BiasChart";
import { MetricsCard } from "@/components/MetricsCard";
import { SuggestionBox } from "@/components/SuggestionBox";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog } from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { AnalyzeResponse } from "@/lib/types";

export default function ResultsPage() {
  const [data, setData] = useState<AnalyzeResponse | null>(null);
  const [showDialog, setShowDialog] = useState(false);

  useEffect(() => {
    const raw = localStorage.getItem("analysisResult");
    if (!raw) {
      return;
    }

    try {
      const parsed = JSON.parse(raw) as AnalyzeResponse;
      setData(parsed);
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

  if (!data) {
    return (
      <main>
        <div className="container">
          <h1 className="text-2xl font-bold">No Analysis Result Found</h1>
          <p className="muted">Run an analysis first from the upload page.</p>
          <Link href="/" className="mt-4 inline-block text-sm text-slate-700 underline">
            Back to Upload
          </Link>
        </div>
      </main>
    );
  }

  return (
    <main>
      <div className="container grid" style={{ gap: 16 }}>
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold">Bias Analysis Dashboard</h1>
          <Badge variant={data.bias ? "destructive" : "success"}>
            {data.bias ? "Bias Detected" : "No Severe Bias"}
          </Badge>
        </div>

        <div className="grid grid-3">
          <MetricsCard label="Demographic Parity Diff" value={data.dp_diff.toFixed(4)} />
          <MetricsCard label="Equalized Odds Diff" value={data.eo_diff.toFixed(4)} />
          <MetricsCard label="Intent" value={data.intent} helper={`Domain: ${data.domain}`} />
        </div>

        <BiasChart dpDiff={data.dp_diff} eoDiff={data.eo_diff} />

        <Card>
          <CardHeader>
            <CardTitle>Group Metrics</CardTitle>
          </CardHeader>
          <CardContent>
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
          </CardContent>
        </Card>

        <SuggestionBox suggestions={data.suggestions} />

        <div className="flex items-center gap-3">
          <Button onClick={() => setShowDialog(true)}>Explain</Button>
          <Link href="/" className="text-sm text-slate-700 underline">
            Run another analysis
          </Link>
        </div>

        <Dialog open={showDialog} onOpenChange={setShowDialog} title="LLM Explanation">
          <div className="grid" style={{ gap: 12 }}>
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
      </div>
    </main>
  );
}

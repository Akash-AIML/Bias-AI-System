"use client";

import { useEffect } from "react";
import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Info } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Select } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import type { AnalyzeResponse } from "@/lib/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export function FileUpload() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [target, setTarget] = useState("");
  const [sensitive, setSensitive] = useState("");
  const [predictionColumn, setPredictionColumn] = useState("");
  const [query, setQuery] = useState("check bias");
  const [isLoading, setIsLoading] = useState(false);
  const [isParsing, setIsParsing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const selectableColumns = useMemo(() => columns, [columns]);

  useEffect(() => {
    if (!isLoading) {
      setProgress(0);
      return;
    }

    const id = window.setInterval(() => {
      setProgress((prev) => (prev >= 90 ? 90 : prev + 10));
    }, 250);

    return () => window.clearInterval(id);
  }, [isLoading]);

  async function parseColumnsFromFile(selectedFile: File) {
    setIsParsing(true);
    const text = await selectedFile.text();
    localStorage.setItem("analysisFileContent", text);
    const firstLine = text.split("\n")[0] ?? "";
    const parsedColumns = firstLine
      .split(",")
      .map((column) => column.trim().replaceAll('"', ""))
      .filter(Boolean);

    setColumns(parsedColumns);
    setTarget(parsedColumns[0] ?? "");
    setSensitive(parsedColumns[1] ?? "");
    setPredictionColumn("");
    setIsParsing(false);
  }

  async function handleAnalyze() {
    if (!file) {
      setError("Please upload a CSV file.");
      return;
    }
    if (!target || !sensitive) {
      setError("Please select both target and sensitive columns.");
      return;
    }

    setError(null);
    setIsLoading(true);
    setProgress(20);

    try {
      const payload = new FormData();
      payload.append("file", file);
      payload.append("target", target);
      payload.append("sensitive", sensitive);
      payload.append("query", query || "check bias");
      if (predictionColumn) {
        payload.append("prediction_column", predictionColumn);
      }

      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: payload,
      });

      if (!response.ok) {
        const details = await response.json().catch(() => ({}));
        throw new Error(details?.detail ?? "Analysis request failed.");
      }

      const result = (await response.json()) as AnalyzeResponse;
      setProgress(100);
      localStorage.setItem("analysisResult", JSON.stringify(result));
      localStorage.setItem(
        "analysisMeta",
        JSON.stringify({ filename: file.name, target, sensitive, predictionColumn, query })
      );
      router.push("/results");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unexpected error");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <TooltipProvider>
      <div className="grid gap-6">
        <Card className="animate-fade-rise">
          <CardHeader>
            <CardTitle>Upload</CardTitle>
            <p className="text-sm text-[var(--text-soft)]">Select a CSV to initialize columns and start configuration.</p>
          </CardHeader>
          <CardContent className="space-y-4">
            <Input
              type="file"
              accept=".csv"
              onChange={async (event) => {
                const selectedFile = event.target.files?.[0] ?? null;
                setFile(selectedFile);
                setColumns([]);
                if (selectedFile) {
                  await parseColumnsFromFile(selectedFile);
                }
              }}
            />

            {isParsing ? (
              <div className="space-y-2">
                <Skeleton className="h-9 w-full" />
                <Skeleton className="h-9 w-full" />
              </div>
            ) : null}

            {file ? (
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <Badge variant="secondary">{file.name}</Badge>
                <Badge variant="secondary">{columns.length} columns found</Badge>
              </div>
            ) : null}
          </CardContent>
        </Card>

        <Separator />

        <Card className="animate-fade-rise">
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
            <p className="text-sm text-[var(--text-soft)]">Choose prediction target, sensitive attribute, and optional analysis query.</p>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <div className="mb-1 flex items-center gap-2 text-sm font-medium text-[var(--text)]">
                  <span>Target Column</span>
                  <Tooltip>
                    <TooltipTrigger>
                      <span tabIndex={0} className="inline-flex rounded text-slate-500">
                        <Info size={14} />
                      </span>
                      <TooltipContent>
                        Pick the outcome column you want to audit, usually a decision label such as approved/rejected.
                      </TooltipContent>
                    </TooltipTrigger>
                  </Tooltip>
                </div>
                <Select value={target} onChange={(event) => setTarget(event.target.value)}>
                  <option value="">Select target column</option>
                  {selectableColumns.map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
                </Select>
              </div>

              <div>
                <div className="mb-1 flex items-center gap-2 text-sm font-medium text-[var(--text)]">
                  <span>Sensitive Attribute</span>
                  <Tooltip>
                    <TooltipTrigger>
                      <span tabIndex={0} className="inline-flex rounded text-slate-500">
                        <Info size={14} />
                      </span>
                      <TooltipContent>
                        Pick the protected group column used for fairness comparison, such as gender or region.
                      </TooltipContent>
                    </TooltipTrigger>
                  </Tooltip>
                </div>
                <Select value={sensitive} onChange={(event) => setSensitive(event.target.value)}>
                  <option value="">Select sensitive column</option>
                  {selectableColumns.map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
                </Select>
              </div>
            </div>

            <div>
              <div className="mb-1 flex items-center gap-2 text-sm font-medium text-[var(--text)]">
                <span>Prediction Column (optional)</span>
                <Tooltip>
                  <TooltipTrigger>
                    <span tabIndex={0} className="inline-flex rounded text-slate-500">
                      <Info size={14} />
                    </span>
                    <TooltipContent>
                      Provide model predictions to audit real outputs. Leave blank to run a proxy audit.
                    </TooltipContent>
                  </TooltipTrigger>
                </Tooltip>
              </div>
              <Select value={predictionColumn} onChange={(event) => setPredictionColumn(event.target.value)}>
                <option value="">No prediction column</option>
                {selectableColumns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </Select>
            </div>

            <div>
              <label className="mb-1 block text-sm font-medium text-[var(--text)]">Optional Query</label>
              <Textarea
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="check bias in selection decisions"
              />
            </div>

            {error ? (
              <Alert variant="destructive">
                <AlertTitle>Analysis warning</AlertTitle>
                <AlertDescription>
                  {error} Try selecting a binary/categorical target column or a column with enough samples per class.
                </AlertDescription>
              </Alert>
            ) : null}

            {isLoading ? (
              <div className="space-y-2">
                <div className="text-xs text-[var(--text-soft)]">Analysis in progress</div>
                <Progress value={progress} />
              </div>
            ) : null}

            <Button onClick={handleAnalyze} disabled={isLoading} className="w-full" size="lg">
              {isLoading ? "Analyzing..." : "Analyze"}
            </Button>
          </CardContent>
        </Card>

        <Separator />

        <Card className="animate-fade-rise">
          <CardHeader>
            <CardTitle>Results Preview</CardTitle>
            <p className="text-sm text-[var(--text-soft)]">Preview current selection before running the fairness pipeline.</p>
          </CardHeader>
          <CardContent>
            {!file ? (
              <div className="space-y-2">
                <Skeleton className="h-4 w-2/3" />
                <Skeleton className="h-4 w-1/2" />
              </div>
            ) : (
              <div className="grid gap-2 text-sm text-[var(--text)]">
                <div>
                  <span className="font-medium">Dataset:</span> {file.name}
                </div>
                <div>
                  <span className="font-medium">Target:</span> {target || "Not selected"}
                </div>
                <div>
                  <span className="font-medium">Sensitive:</span> {sensitive || "Not selected"}
                </div>
                <div>
                  <span className="font-medium">Predictions:</span> {predictionColumn || "Proxy audit"}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </TooltipProvider>
  );
}

"use client";

import { useEffect, useMemo, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { UploadCloud, Database, Target, Shield, Info, ArrowRight, Settings2, CheckCircle2, Play } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { setAnalysisFile } from "@/lib/analysis-cache";
import type { AnalyzeResponse } from "@/lib/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api";

type ColumnSuggestionResponse = {
  columns: string[];
  target: string | null;
  sensitive: string | null;
  prediction_column: string | null;
  time_column: string | null;
  method: string;
  notes: string[];
};

function parseCsvHeader(line: string): string[] {
  const columns: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    const nextCharacter = line[index + 1];

    if (character === '"') {
      if (inQuotes && nextCharacter === '"') {
        current += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (character === "," && !inQuotes) {
      const trimmed = current.trim();
      if (trimmed) {
        columns.push(trimmed);
      }
      current = "";
      continue;
    }

    current += character;
  }

  const trimmed = current.trim();
  if (trimmed) {
    columns.push(trimmed);
  }

  return columns;
}

export function FileUpload() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [isHovering, setIsHovering] = useState(false);

  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [target, setTarget] = useState("");
  const [sensitive, setSensitive] = useState("");
  const [predictionColumn, setPredictionColumn] = useState("");
  const [weightColumn, setWeightColumn] = useState("");
  const [orgName, setOrgName] = useState("");
  const [datasetName, setDatasetName] = useState("");
  const [timeColumn, setTimeColumn] = useState("");
  const [textColumns, setTextColumns] = useState("");
  const [targetBinarizationThreshold, setTargetBinarizationThreshold] = useState("");
  const [query, setQuery] = useState("check bias");

  const [isLoading, setIsLoading] = useState(false);
  const [isParsing, setIsParsing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [autoMappingNote, setAutoMappingNote] = useState<string | null>(null);

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
    // Frontend guard: reject files over 50 MB immediately (backend limit is same)
    const MAX_FILE_BYTES = 50 * 1024 * 1024;
    if (selectedFile.size > MAX_FILE_BYTES) {
      setError(
        `File is too large (${(selectedFile.size / 1024 / 1024).toFixed(1)} MB). ` +
        `Maximum supported size is 50 MB. Please reduce the dataset size or sample it down.`
      );
      return;
    }

    setIsParsing(true);
    try {
      const text = await selectedFile.text();
      setAnalysisFile(selectedFile);
      const firstLine = text.split("\n")[0] ?? "";
      const parsedColumns = parseCsvHeader(firstLine);

      setColumns(parsedColumns);
      setTarget("");
      setSensitive("");
      setPredictionColumn("");
      setTimeColumn("");
      setTextColumns("");
      setFile(selectedFile);

      try {
        const suggestionPayload = new FormData();
        suggestionPayload.append("file", selectedFile);
        const suggestionResponse = await fetch(`${API_BASE_URL}/column-suggestions`, {
          method: "POST",
          body: suggestionPayload,
        });

        if (suggestionResponse.ok) {
          const suggestions = (await suggestionResponse.json()) as ColumnSuggestionResponse;
          setColumns(suggestions.columns?.length ? suggestions.columns : parsedColumns);
          setTarget(suggestions.target ?? "");
          setSensitive(suggestions.sensitive ?? "");
          setPredictionColumn(suggestions.prediction_column ?? "");
            setWeightColumn((suggestions as any).weight_column ?? "");
          setTimeColumn(suggestions.time_column ?? "");
          setAutoMappingNote(
            suggestions.notes?.length
              ? `Auto-mapped using ${suggestions.method}: ${suggestions.notes.join(" ")}`
              : `Auto-mapped using ${suggestions.method}.`
          );
        } else {
          setAutoMappingNote("Auto-mapping could not run; please confirm target and sensitive columns.");
        }
      } catch {
        setAutoMappingNote("Auto-mapping could not run; please confirm target and sensitive columns.");
      }

      setStep(2); // Move to next step automatically
    } catch {
      setError("Failed to parse file. Ensure it is a valid CSV.");
    } finally {
      setIsParsing(false);
    }
  }

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsHovering(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.name.endsWith(".csv")) {
      await parseColumnsFromFile(droppedFile);
    } else {
      setError("Please drop a valid .csv file.");
    }
  };

  async function handleAnalyze() {
    if (!file) {
      setError("Please upload a CSV file.");
      setStep(1);
      return;
    }
    if (!target || !sensitive) {
      setError("Please select both target and sensitive columns.");
      setStep(2);
      return;
    }
    if (predictionColumn && (predictionColumn === target || predictionColumn === sensitive)) {
      setError("Prediction column must be different from target and sensitive columns.");
      setStep(2);
      return;
    }
    if (targetBinarizationThreshold.trim() && Number.isNaN(Number(targetBinarizationThreshold.trim()))) {
      setError("Target threshold must be a valid number.");
      setStep(3);
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
      if (predictionColumn) payload.append("prediction_column", predictionColumn);
      if (weightColumn) payload.append("weight_column", weightColumn);
      if (orgName.trim()) payload.append("org_name", orgName.trim());
      if (datasetName.trim()) payload.append("dataset_name", datasetName.trim());
      if (timeColumn) payload.append("time_column", timeColumn);
      if (textColumns.trim()) payload.append("text_columns", textColumns.trim());
      if (targetBinarizationThreshold.trim()) {
        payload.append("target_binarization_threshold", targetBinarizationThreshold.trim());
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
        JSON.stringify({
          filename: file.name,
          target,
          sensitive,
          predictionColumn,
          weightColumn,
          query,
          orgName,
          datasetName,
          timeColumn,
          textColumns,
          targetBinarizationThreshold,
          uploadedAt: new Date().toISOString(),
        })
      );
      router.push("/results");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unexpected error");
      setIsLoading(false);
    }
  }

  const renderColumnSelector = (label: string, icon: React.ReactNode, value: string, setter: (val: string) => void, tooltip: string) => {
    return (
      <Card className="overflow-hidden border border-slate-200/50 shadow-sm transition-all hover:border-slate-300 dark:border-slate-800 dark:hover:border-slate-700">
        <CardContent className="p-4 sm:p-6">
          <div className="mb-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400">
                {icon}
              </div>
              <h3 className="font-semibold text-[var(--text)]">{label}</h3>
              <Tooltip>
                <TooltipTrigger>
                  <Info size={14} className="text-slate-400 hover:text-slate-600" />
                </TooltipTrigger>
                <TooltipContent>{tooltip}</TooltipContent>
              </Tooltip>
            </div>
            {value && <CheckCircle2 size={18} className="text-emerald-500" />}
          </div>
          <div className="relative">
            <select
              title={`Select ${label}`}
              value={value}
              onChange={(e) => setter(e.target.value)}
              className="w-full appearance-none rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-[var(--text)] outline-none focus:border-slate-400 focus:ring-1 focus:ring-slate-400 dark:border-slate-800 dark:bg-slate-900 dark:focus:border-slate-600 dark:focus:ring-slate-600"
            >
              <option value="">Choose a column...</option>
              {selectableColumns.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
            <div className="pointer-events-none absolute right-3 top-2.5 text-slate-400">
              <ArrowRight size={14} className="rotate-90" />
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <TooltipProvider>
      <div className="mx-auto w-full max-w-4xl animate-fade-in space-y-8">

        {/* Progress Stepper */}
        <div className="flex items-center justify-between px-4">
          <div className={`flex flex-col items-center gap-2 ${step >= 1 ? "text-[var(--text)]" : "text-slate-400"}`}>
            <div className={`flex h-8 w-8 items-center justify-center rounded-full ${step >= 1 ? "bg-slate-900 text-white dark:bg-white dark:text-black" : "bg-slate-100"}`}>1</div>
            <span className="text-xs font-medium uppercase tracking-wider">Upload</span>
          </div>
          <div className={`h-px flex-1 mx-4 ${step >= 2 ? "bg-slate-900 dark:bg-white" : "bg-slate-200 dark:bg-slate-800"}`} />
          <div className={`flex flex-col items-center gap-2 ${step >= 2 ? "text-[var(--text)]" : "text-slate-400"}`}>
            <div className={`flex h-8 w-8 items-center justify-center rounded-full ${step >= 2 ? "bg-slate-900 text-white dark:bg-white dark:text-black" : "bg-slate-100 dark:bg-slate-800"}`}>2</div>
            <span className="text-xs font-medium uppercase tracking-wider">Map Data</span>
          </div>
          <div className={`h-px flex-1 mx-4 ${step >= 3 ? "bg-slate-900 dark:bg-white" : "bg-slate-200 dark:bg-slate-800"}`} />
          <div className={`flex flex-col items-center gap-2 ${step >= 3 ? "text-[var(--text)]" : "text-slate-400"}`}>
            <div className={`flex h-8 w-8 items-center justify-center rounded-full ${step >= 3 ? "bg-slate-900 text-white dark:bg-white dark:text-black" : "bg-slate-100 dark:bg-slate-800"}`}>3</div>
            <span className="text-xs font-medium uppercase tracking-wider">Configure</span>
          </div>
        </div>

        {error ? (
          <Alert variant="destructive" className="animate-fade-in border-red-200 bg-red-50 text-red-900 dark:border-red-900/50 dark:bg-red-900/10 dark:text-red-200">
            <AlertTitle className="font-semibold">Configuration Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : null}
        {autoMappingNote ? (
          <Alert className="animate-fade-in">
            <AlertTitle className="font-semibold">Auto Mapping</AlertTitle>
            <AlertDescription>{autoMappingNote}</AlertDescription>
          </Alert>
        ) : null}

        <div className="relative min-h-[400px]">
          {/* STEP 1: UPLOAD */}
          {step === 1 && (
            <div className="animate-fade-in w-full">
              <div
                className={`flex min-h-[400px] flex-col items-center justify-center rounded-xl border-2 border-dashed transition-all duration-200 ${isHovering ? "border-slate-400 bg-slate-50 dark:border-slate-500 dark:bg-slate-900/50" : "border-slate-200 bg-white/50 dark:border-slate-800 dark:bg-black/20"
                  }`}
                onDragOver={(e) => { e.preventDefault(); setIsHovering(true); }}
                onDragLeave={() => setIsHovering(false)}
                onDrop={handleDrop}
              >
                <div className="flex h-20 w-20 items-center justify-center rounded-full bg-slate-100 text-slate-500 mb-6 dark:bg-slate-900 dark:text-slate-400">
                  <UploadCloud size={40} strokeWidth={1.5} />
                </div>
                <h2 className="text-xl font-semibold tracking-tight text-[var(--text)]">Upload your dataset</h2>
                <p className="mt-2 text-center text-sm text-[var(--text-soft)] max-w-sm">
                  Drag and drop your CSV file here, or click to browse. We&apos;ll automatically parse your columns for auditing.
                </p>

                <input
                  type="file"
                  accept=".csv"
                  className="hidden"
                  ref={fileInputRef}
                  onChange={async (e) => {
                    const f = e.target.files?.[0];
                    if (f) await parseColumnsFromFile(f);
                  }}
                />
                <Button
                  onClick={() => fileInputRef.current?.click()}
                  className="mt-8 px-8"
                  variant="outline"
                  disabled={isParsing}
                >
                  {isParsing ? "Reading Data..." : "Browse Files"}
                </Button>
              </div>
            </div>
          )}

          {/* STEP 2: MAP DATA */}
          {step === 2 && (
            <div className="animate-fade-in space-y-6 w-full">
              <div className="flex items-center justify-between rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-950">
                <div className="flex items-center gap-4">
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-emerald-100 text-emerald-600 dark:bg-emerald-900/30 dark:text-emerald-400">
                    <Database size={20} />
                  </div>
                  <div>
                    <h3 className="font-medium text-[var(--text)]">{file?.name}</h3>
                    <p className="text-xs text-[var(--text-soft)]">{columns.length} columns detected</p>
                  </div>
                </div>
                <Button variant="ghost" size="sm" onClick={() => setStep(1)} className="text-xs">Change File</Button>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                {renderColumnSelector(
                  "Target Column",
                  <Target size={16} />,
                  target,
                  setTarget,
                  "Pick the outcome column you want to audit (e.g. approved/rejected)."
                )}
                {renderColumnSelector(
                  "Sensitive Attribute",
                  <Shield size={16} />,
                  sensitive,
                  setSensitive,
                  "Protected group column used for fairness comparison (e.g. gender/race)."
                )}
                {renderColumnSelector(
                  "Prediction (Optional)",
                  <Play size={16} />,
                  predictionColumn,
                  setPredictionColumn,
                  "Model predictions to audit real outputs. Leave blank for proxy audit."
                )}
                {renderColumnSelector(
                  "Weight Column (Optional)",
                  <Database size={16} />,
                  weightColumn,
                  setWeightColumn,
                  "Optional sample weight column to apply per-row weights during analysis."
                )}
                {renderColumnSelector(
                  "Time Feature (Optional)",
                  <Settings2 size={16} />,
                  timeColumn,
                  setTimeColumn,
                  "Column indicating time, for temporal drift analysis."
                )}
              </div>

              <div className="flex justify-end pt-4">
                <Button onClick={() => setStep(3)} className="gap-2" disabled={!target || !sensitive}>
                  Continue to Configuration <ArrowRight size={16} />
                </Button>
              </div>
            </div>
          )}

          {/* STEP 3: CONFIGURE & RUN */}
          {step === 3 && (
            <div className="animate-fade-in rounded-xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950 w-full">
              <h2 className="mb-6 text-xl font-semibold tracking-tight text-[var(--text)]">Finalize Audit Parameters</h2>

              <div className="space-y-6">
                <div className="grid gap-6 sm:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-[var(--text)]">Organization Name</label>
                    <Input value={orgName} onChange={(e) => setOrgName(e.target.value)} placeholder="Acme Corp" className="h-11" />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-[var(--text)]">Dataset Name</label>
                    <Input value={datasetName} onChange={(e) => setDatasetName(e.target.value)} placeholder="Q1 Hiring Decisions" className="h-11" />
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-[var(--text)]">Audit Goal / Context Query</label>
                  <Textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="E.g., Ensure our resume screening process doesn't disadvantage specific demographics..."
                    className="min-h-[100px] resize-none"
                  />
                </div>

                {isLoading && (
                  <div className="space-y-3 rounded-lg border border-slate-200 bg-slate-50 p-4 dark:border-slate-800 dark:bg-slate-900/50">
                    <div className="flex justify-between text-sm font-medium">
                      <span>Analyzing dataset and generating narrative...</span>
                      <span>{progress}%</span>
                    </div>
                    <Progress value={progress} className="h-2" />
                  </div>
                )}
              </div>

              <div className="mt-8 flex justify-between">
                <Button variant="ghost" onClick={() => setStep(2)}>Back</Button>
                <Button onClick={handleAnalyze} disabled={isLoading} size="lg" className="min-w-[160px]">
                  {isLoading ? "Auditing..." : "Run Complete Audit"}
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </TooltipProvider>
  );
}

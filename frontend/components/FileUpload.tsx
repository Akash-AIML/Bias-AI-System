"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import type { AnalyzeResponse } from "@/lib/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export function FileUpload() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [target, setTarget] = useState("");
  const [sensitive, setSensitive] = useState("");
  const [query, setQuery] = useState("check bias");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectableColumns = useMemo(() => columns, [columns]);

  async function parseColumnsFromFile(selectedFile: File) {
    const text = await selectedFile.text();
    const firstLine = text.split("\n")[0] ?? "";
    const parsedColumns = firstLine
      .split(",")
      .map((column) => column.trim().replaceAll('"', ""))
      .filter(Boolean);

    setColumns(parsedColumns);
    setTarget(parsedColumns[0] ?? "");
    setSensitive(parsedColumns[1] ?? "");
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

    try {
      const payload = new FormData();
      payload.append("file", file);
      payload.append("target", target);
      payload.append("sensitive", sensitive);
      payload.append("query", query || "check bias");

      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: payload,
      });

      if (!response.ok) {
        const details = await response.json().catch(() => ({}));
        throw new Error(details?.detail ?? "Analysis request failed.");
      }

      const result = (await response.json()) as AnalyzeResponse;
      localStorage.setItem("analysisResult", JSON.stringify(result));
      localStorage.setItem(
        "analysisMeta",
        JSON.stringify({ filename: file.name, target, sensitive, query })
      );
      router.push("/results");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unexpected error");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Upload Dataset for Fairness Analysis</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid" style={{ gap: 12 }}>
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

          <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div>
              <label className="mb-1 block text-sm font-medium">Target Column</label>
              <Select value={target} onChange={(event) => setTarget(event.target.value)}>
                {selectableColumns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </Select>
            </div>

            <div>
              <label className="mb-1 block text-sm font-medium">Sensitive Attribute</label>
              <Select value={sensitive} onChange={(event) => setSensitive(event.target.value)}>
                {selectableColumns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </Select>
            </div>
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium">Optional Query</label>
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="check bias"
            />
          </div>

          {error ? <div className="text-sm" style={{ color: "#b91c1c" }}>{error}</div> : null}

          <Button onClick={handleAnalyze} disabled={isLoading}>
            {isLoading ? "Analyzing..." : "Analyze"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

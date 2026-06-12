let analysisFile: File | null = null;

export function setAnalysisFile(file: File | null): void {
  analysisFile = file;
}

export function getAnalysisFile(): File | null {
  return analysisFile;
}

export function clearAnalysisFile(): void {
  analysisFile = null;
}
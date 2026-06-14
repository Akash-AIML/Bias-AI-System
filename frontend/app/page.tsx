import { FileUpload } from "@/components/FileUpload";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { ThemeToggle } from "@/components/ThemeToggle";

export default function HomePage() {
  return (
    <main>
      <div className="mx-auto mb-10 max-w-5xl px-4 py-5 sm:px-6 lg:px-8">
        <div className="mb-6 flex items-start justify-between gap-4">
          <div className="space-y-3 mt-5">
            <h1 className="text-5xl font-semibold text-[var(--text)] animate-fade-rise">
              AI Audit System
            </h1>
            <p className="max-w-2xl text-sm text-[var(--text-soft)] animate-fade-rise">
              Upload a CSV, configure inputs, and generate compliance-ready findings.
            </p>
          </div>
        </div>

        <Card className="animate-fade-rise">
          <CardHeader>
            <CardTitle className="text-xl">Audit workflow</CardTitle>
            <p className="text-sm text-[var(--text-soft)]">
              Structured upload, configuration, status, and preview zones.
            </p>
          </CardHeader>
          <CardContent className="space-y-6">
            <Separator />
            <FileUpload />
          </CardContent>
        </Card>
      </div>
    </main>
  );
}

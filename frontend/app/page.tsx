import { FileUpload } from "@/components/FileUpload";

export default function HomePage() {
  return (
    <main>
      <div className="container">
        <h1 className="mb-2 text-3xl font-bold">AI Fairness Audit System</h1>
        <p className="muted mb-6">
          Upload a CSV, select target and sensitive columns, and analyze bias with explainable results.
        </p>
        <FileUpload />
      </div>
    </main>
  );
}

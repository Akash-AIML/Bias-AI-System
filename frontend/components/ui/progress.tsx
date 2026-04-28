import * as React from "react";

import { cn } from "@/lib/utils";

function Progress({ value = 0, className }: { value?: number; className?: string }) {
  const clamped = Math.max(0, Math.min(100, value));

  return (
    <div className={cn("relative h-2 w-full overflow-hidden rounded-full", className)} style={{ background: "var(--surface-muted)" }}>
      <div
        className="h-full transition-all duration-500 ease-out"
        style={{ width: `${clamped}%`, background: "linear-gradient(90deg, var(--brand), var(--accent))" }}
      />
    </div>
  );
}

export { Progress };

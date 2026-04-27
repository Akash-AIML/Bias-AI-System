import * as React from "react";

import { cn } from "@/lib/utils";

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "destructive" | "success" | "neutral";
}

function Badge({ className, variant = "neutral", ...props }: BadgeProps) {
  return (
    <div
      className={cn(
        "inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold",
        variant === "destructive" && "border-red-200 bg-red-50 text-red-700",
        variant === "success" && "border-emerald-200 bg-emerald-50 text-emerald-700",
        variant === "neutral" && "border-slate-200 bg-slate-50 text-slate-700",
        className
      )}
      {...props}
    />
  );
}

export { Badge };

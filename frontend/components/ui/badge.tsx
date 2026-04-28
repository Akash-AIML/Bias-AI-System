import * as React from "react"
import { cn } from "@/lib/utils"

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "secondary" | "destructive" | "success"
}

function Badge({ className, variant = "default", ...props }: BadgeProps) {
  return (
    <div
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-3 py-1 text-xs font-semibold transition-colors",
        variant === "default" && "",
        variant === "secondary" && "",
        variant === "destructive" && "",
        variant === "success" && "",
        className
      )}
      style={{
        borderColor:
          variant === "destructive"
            ? "var(--danger)"
            : variant === "success"
              ? "var(--success)"
              : "var(--border)",
        background:
          variant === "destructive"
            ? "var(--danger-soft)"
            : variant === "success"
              ? "var(--success-soft)"
              : "var(--surface-muted)",
        color:
          variant === "destructive"
            ? "var(--danger)"
            : variant === "success"
              ? "var(--success)"
              : "var(--text-soft)",
      }}
      {...props}
    />
  )
}

export { Badge }

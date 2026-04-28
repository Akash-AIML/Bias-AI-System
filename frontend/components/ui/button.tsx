import * as React from "react"
import { cn } from "@/lib/utils"

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "outline" | "ghost" | "destructive"
  size?: "default" | "sm" | "lg"
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "default", ...props }, ref) => (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-md font-medium transition-all disabled:pointer-events-none disabled:opacity-50",
        variant === "default" && "text-white shadow-sm hover:scale-[1.01] hover:shadow-md focus:outline-none focus:ring-2 focus:ring-offset-2",
        variant === "outline" && "border bg-transparent hover:scale-[1.01]",
        variant === "ghost" && "hover:scale-[1.01]",
        variant === "destructive" && "text-white hover:scale-[1.01]",
        size === "default" && "h-10 px-4 py-2 text-sm",
        size === "sm" && "h-8 px-3 text-xs",
        size === "lg" && "h-12 px-6 text-base",
        className
      )}
      style={{
        background:
          variant === "default"
            ? "var(--brand)"
            : variant === "destructive"
              ? "var(--danger)"
              : "transparent",
        borderColor: variant === "outline" ? "var(--border)" : undefined,
        color:
          variant === "default" || variant === "destructive"
            ? "#fff"
            : "var(--text)",
      }}
      ref={ref}
      {...props}
    />
  )
)
Button.displayName = "Button"

export { Button }

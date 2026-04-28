"use client";

import * as React from "react";

import { cn } from "@/lib/utils";

function TooltipProvider({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}

function Tooltip({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}

function TooltipTrigger({ children }: { children: React.ReactNode }) {
  return <span className="group relative inline-flex">{children}</span>;
}

function TooltipContent({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <span
      className={cn(
        "pointer-events-none absolute left-1/2 top-full z-30 mt-2 hidden w-64 -translate-x-1/2 rounded-md border border-slate-200 bg-white p-2 text-xs text-slate-700 shadow-md group-hover:block group-focus-within:block",
        className
      )}
    >
      {children}
    </span>
  );
}

export { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider };

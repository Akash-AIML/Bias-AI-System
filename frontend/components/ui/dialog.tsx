"use client";

import { ReactNode } from "react";

import { cn } from "@/lib/utils";

type DialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  children: ReactNode;
};

export function Dialog({ open, onOpenChange, title, children }: DialogProps) {
  if (!open) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(12, 17, 29, 0.58)" }}
      onClick={() => onOpenChange(false)}
    >
      <div
        className={cn("w-full max-w-2xl rounded-2xl border p-6 shadow-2xl animate-fade-rise")}
        style={{ borderColor: "var(--border)", background: "var(--surface)", color: "var(--text)" }}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h3 className="m-0 text-lg font-semibold">{title}</h3>
          <button
            onClick={() => onOpenChange(false)}
            className="rounded-md border px-2 py-1 text-sm"
            style={{ borderColor: "var(--border)", background: "var(--surface-muted)" }}
            type="button"
          >
            Close
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

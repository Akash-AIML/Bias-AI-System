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
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4"
      onClick={() => onOpenChange(false)}
    >
      <div
        className={cn("w-full max-w-2xl rounded-xl border border-slate-200 bg-white p-6")}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h3 className="m-0 text-lg font-semibold">{title}</h3>
          <button
            onClick={() => onOpenChange(false)}
            className="rounded-md border border-slate-300 px-2 py-1 text-sm"
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

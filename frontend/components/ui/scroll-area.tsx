import * as React from "react";

import { cn } from "@/lib/utils";

function ScrollArea({ className, children }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("relative overflow-auto", className)}>{children}</div>;
}

export { ScrollArea };

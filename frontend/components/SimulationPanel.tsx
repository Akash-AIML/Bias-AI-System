import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

import type { MitigationSimulation } from "@/lib/types";
import { ScrollArea } from "./ui/scroll-area";

type SimulationPanelProps = {
  simulations: MitigationSimulation[];
};

export function SimulationPanel({ simulations }: SimulationPanelProps) {
  const rows = simulations;

  return (
    <Card className="animate-fade-rise">
      <CardHeader>
        <CardTitle>Before/After Simulation</CardTitle>
        <p className="text-sm text-[var(--text-soft)]">
          Uses the backend mitigation simulation results for the current audit.
        </p>
      </CardHeader>
      <CardContent>
        <ScrollArea className="max-h-64">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Method</TableHead>
              <TableHead>BSI</TableHead>
              <TableHead>DP Diff</TableHead>
              <TableHead>EO Diff</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.name}>
                <TableCell>{row.name}</TableCell>
                <TableCell>{row.bsi_score.toFixed(2)}</TableCell>
                <TableCell>{row.dp_diff.toFixed(4)}</TableCell>
                <TableCell>{row.eo_diff.toFixed(4)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
          </Table>
          </ScrollArea>
      </CardContent>
    </Card>
  );
}

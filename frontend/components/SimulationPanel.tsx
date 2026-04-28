import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

import type { GroupMetric } from "@/lib/types";

type SimulationPanelProps = {
  metrics: Record<string, GroupMetric>;
};

export function SimulationPanel({ metrics }: SimulationPanelProps) {
  const entries = Object.entries(metrics);
  const selectionRates = entries.map(([, values]) => values.selection_rate);
  const mean = selectionRates.reduce((sum, value) => sum + value, 0) / (selectionRates.length || 1);

  const rows = entries.map(([group, values]) => {
    const projected = values.selection_rate - (values.selection_rate - mean) * 0.5;
    return {
      group,
      current: values.selection_rate,
      projected,
    };
  });

  return (
    <Card className="animate-fade-rise">
      <CardHeader>
        <CardTitle>Before/After Simulation</CardTitle>
        <p className="text-sm text-[var(--text-soft)]">
          Simulates a 50% reduction in demographic parity gap by moving selection rates toward the mean.
        </p>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Group</TableHead>
              <TableHead>Current Rate</TableHead>
              <TableHead>Projected Rate</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.group}>
                <TableCell>{row.group}</TableCell>
                <TableCell>{row.current.toFixed(3)}</TableCell>
                <TableCell>{row.projected.toFixed(3)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

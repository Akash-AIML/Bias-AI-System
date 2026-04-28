"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { GroupMetric } from "@/lib/types";

type GroupMetricsChartProps = {
  metrics: Record<string, GroupMetric>;
};

export function GroupMetricsChart({ metrics }: GroupMetricsChartProps) {
  const data = Object.entries(metrics).map(([group, values]) => ({
    group,
    accuracy: values.accuracy,
    selection_rate: values.selection_rate,
  }));

  return (
    <Card className="animate-fade-rise">
      <CardHeader>
        <CardTitle>Group Comparison</CardTitle>
        <p className="text-sm text-[var(--text-soft)]">Accuracy and selection rate by group.</p>
      </CardHeader>
      <CardContent>
        <div style={{ width: "100%", height: 280 }}>
          <ResponsiveContainer>
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
              <XAxis dataKey="group" stroke="var(--text-soft)" />
              <YAxis domain={[0, 1]} stroke="var(--text-soft)" />
              <Tooltip
                contentStyle={{
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                  borderRadius: 12,
                  color: "var(--text)",
                }}
              />
              <Legend />
              <Bar dataKey="accuracy" fill="var(--brand)" radius={[6, 6, 0, 0]} />
              <Bar dataKey="selection_rate" fill="var(--accent)" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

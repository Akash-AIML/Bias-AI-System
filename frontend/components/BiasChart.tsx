"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

type BiasChartProps = {
  dpDiff: number;
  eoDiff: number;
};

export function BiasChart({ dpDiff, eoDiff }: BiasChartProps) {
  const data = [
    { name: "Demographic Parity", value: Math.abs(dpDiff) },
    { name: "Equalized Odds", value: Math.abs(eoDiff) },
  ];

  return (
    <Card className="animate-fade-rise">
      <CardHeader>
        <CardTitle>Bias Metrics Chart</CardTitle>
        <p className="text-sm text-slate-600">Selection-rate gaps shown as absolute differences.</p>
      </CardHeader>
      <CardContent>
        <div style={{ width: "100%", height: 280 }}>
          <ResponsiveContainer>
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.6} />
              <XAxis dataKey="name" stroke="var(--text-soft)" />
              <YAxis domain={[0, 1]} stroke="var(--text-soft)" />
              <Tooltip
                contentStyle={{
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                  borderRadius: 12,
                  color: "var(--text)",
                }}
              />
              <Bar dataKey="value" fill="var(--brand)" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

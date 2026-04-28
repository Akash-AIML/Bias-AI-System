"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RadialBar, RadialBarChart, ResponsiveContainer } from "recharts";

type GaugeChartProps = {
  label: string;
  value: number;
};

export function GaugeChart({ label, value }: GaugeChartProps) {
  const percent = Math.max(0, Math.min(1, Math.abs(value)));
  const data = [{ name: label, value: percent * 100 }];

  return (
    <Card className="animate-fade-rise">
      <CardHeader>
        <CardTitle>{label}</CardTitle>
        <p className="text-sm text-[var(--text-soft)]">Absolute gap shown on a 0-1 scale.</p>
      </CardHeader>
      <CardContent>
        <div style={{ width: "100%", height: 220 }}>
          <ResponsiveContainer>
            <RadialBarChart
              innerRadius="70%"
              outerRadius="100%"
              data={data}
              startAngle={180}
              endAngle={0}
            >
              <RadialBar dataKey="value" cornerRadius={10} fill="var(--accent)" />
            </RadialBarChart>
          </ResponsiveContainer>
        </div>
        <div className="text-center text-2xl font-semibold text-[var(--text)]">
          {percent.toFixed(2)}
        </div>
      </CardContent>
    </Card>
  );
}

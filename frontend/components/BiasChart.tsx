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
    <Card>
      <CardHeader>
        <CardTitle>Bias Metrics Chart</CardTitle>
      </CardHeader>
      <CardContent>
        <div style={{ width: "100%", height: 280 }}>
          <ResponsiveContainer>
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Bar dataKey="value" fill="#0f172a" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

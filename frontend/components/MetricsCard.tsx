import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type MetricsCardProps = {
  label: string;
  value: string;
  helper?: string;
};

export function MetricsCard({ label, value, helper }: MetricsCardProps) {
  return (
    <Card className="animate-fade-rise">
      <CardHeader>
        <CardTitle className="text-sm font-medium text-slate-600">{label}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-3xl font-bold tracking-tight text-slate-900">{value}</div>
        {helper ? <p className="mt-2 text-xs text-slate-500">{helper}</p> : null}
      </CardContent>
    </Card>
  );
}

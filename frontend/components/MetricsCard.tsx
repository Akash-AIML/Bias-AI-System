import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type MetricsCardProps = {
  label: string;
  value: string;
  helper?: string;
};

export function MetricsCard({ label, value, helper }: MetricsCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{label}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {helper ? <p className="muted mt-2 text-sm">{helper}</p> : null}
      </CardContent>
    </Card>
  );
}

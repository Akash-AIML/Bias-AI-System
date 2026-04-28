import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type SuggestionBoxProps = {
  suggestions: string[];
};

export function SuggestionBox({ suggestions }: SuggestionBoxProps) {
  return (
    <Card className="animate-fade-rise">
      <CardHeader>
        <CardTitle>Mitigation Suggestions</CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="m-0 space-y-2 pl-5">
          {suggestions.map((suggestion) => (
            <li key={suggestion} className="animate-slide-up text-sm leading-6 text-slate-700">
              {suggestion}
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

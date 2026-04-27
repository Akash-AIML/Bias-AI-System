import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type SuggestionBoxProps = {
  suggestions: string[];
};

export function SuggestionBox({ suggestions }: SuggestionBoxProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Mitigation Suggestions</CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="m-0 pl-5">
          {suggestions.map((suggestion) => (
            <li key={suggestion} className="mb-2 text-sm leading-6">
              {suggestion}
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

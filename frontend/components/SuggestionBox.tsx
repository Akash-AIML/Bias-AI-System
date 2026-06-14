import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Sparkles } from "lucide-react";

type SuggestionBoxProps = {
  suggestions: string[];
  onAskGemini?: (suggestion: string) => void;
};

export function SuggestionBox({ suggestions, onAskGemini }: SuggestionBoxProps) {
  return (
    <Card className="animate-fade-rise border border-slate-200 dark:border-slate-800">
      <CardHeader>
        <CardTitle className="text-lg font-semibold flex items-center gap-2 text-slate-900 dark:text-slate-100">
          <Sparkles className="text-indigo-500 w-5 h-5 animate-pulse" />
          Mitigation Suggestions
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {suggestions.map((suggestion, idx) => (
            <div
              key={idx}
              className="animate-slide-up flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 rounded-xl border border-slate-100 dark:border-slate-800 bg-slate-50/30 dark:bg-slate-900/10 hover:border-indigo-500/20 hover:bg-slate-50/70 dark:hover:bg-slate-900/30 transition-all duration-200"
            >
              <span className="text-sm leading-relaxed text-slate-700 dark:text-slate-300 font-medium">
                {suggestion}
              </span>
              {onAskGemini && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onAskGemini(suggestion)}
                  className="shrink-0 self-end sm:self-center flex items-center gap-1.5 h-8 text-[11px] font-semibold border-indigo-200 dark:border-indigo-850 hover:border-indigo-400 bg-white dark:bg-slate-900 hover:bg-indigo-50/30 dark:hover:bg-slate-800 text-indigo-600 dark:text-indigo-400 rounded-lg py-1 px-3 shadow-sm hover:shadow"
                >
                  <Sparkles size={11} className="text-indigo-500" />
                  Ask Gemini
                </Button>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

"use client";

import React, { useState, useEffect, useRef } from "react";
import { Sparkles, Send, X, Bot, User, RefreshCw } from "lucide-react";
import { Sheet, SheetContent } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";

export type AgentMessage = {
  role: "user" | "assistant";
  content: string;
};

type GeminiAgentDrawerProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  auditReport: any;
  messages: AgentMessage[];
  isLoading: boolean;
  onSubmitMessage: (message: string) => Promise<void>;
  onClearHistory: () => void;
};

// Custom lightweight Markdown renderer
function renderMarkdown(text: string) {
  if (!text) return null;
  const blocks = text.split("\n\n");
  return blocks.map((block, index) => {
    // Code blocks
    if (block.startsWith("```")) {
      const lines = block.split("\n");
      const code = lines
        .slice(1, lines.length - (lines[lines.length - 1].startsWith("```") ? 1 : 0))
        .join("\n");
      return (
        <pre
          key={index}
          className="bg-slate-950 text-slate-100 p-3 rounded-lg overflow-x-auto my-2 text-xs font-mono border border-slate-800"
        >
          <code>{code}</code>
        </pre>
      );
    }
    
    // Lists
    if (block.startsWith("- ") || block.startsWith("* ") || /^\d+\.\s/.test(block)) {
      const items = block.split("\n").filter(Boolean);
      const isNumbered = /^\d+\.\s/.test(items[0]);
      const listContent = items.map((item, itemIdx) => {
        const cleanedItem = item.replace(/^(-\s|\*\s|\d+\.\s)/, "");
        return (
          <li key={itemIdx} className="mb-1">
            {parseInlineStyles(cleanedItem)}
          </li>
        );
      });
      return isNumbered ? (
        <ol key={index} className="list-decimal pl-5 my-2 space-y-1 text-sm">
          {listContent}
        </ol>
      ) : (
        <ul key={index} className="list-disc pl-5 my-2 space-y-1 text-sm">
          {listContent}
        </ul>
      );
    }
    
    // Headers
    if (block.startsWith("#")) {
      const level = block.match(/^#+/)?.[0].length ?? 1;
      const content = block.replace(/^#+\s/, "");
      const headerClasses =
        level === 1
          ? "text-lg font-bold my-2 border-b pb-1"
          : level === 2
          ? "text-base font-bold my-2"
          : "text-sm font-semibold my-1";
      const HeaderTag = `h${Math.min(level + 1, 6)}` as any;
      return (
        <HeaderTag key={index} className={headerClasses}>
          {parseInlineStyles(content)}
        </HeaderTag>
      );
    }
    
    // Fallback normal paragraph
    return (
      <p key={index} className="mb-2 leading-relaxed text-sm">
        {parseInlineStyles(block)}
      </p>
    );
  });
}

function parseInlineStyles(text: string) {
  // Support bold (**bold**) and inline code (`code`)
  const parts = text.split(/(\*\*.*?\*\*|`.*?`|\[.*?\]\(.*?\))/g);
  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={index} className="font-semibold text-slate-900 dark:text-slate-100">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code
          key={index}
          className="bg-slate-100 dark:bg-slate-800 text-pink-600 dark:text-pink-400 px-1 py-0.5 rounded text-xs font-mono"
        >
          {part.slice(1, -1)}
        </code>
      );
    }
    if (part.startsWith("[") && part.includes("](")) {
      const match = part.match(/\[(.*?)\]\((.*?)\)/);
      if (match) {
        return (
          <a
            key={index}
            href={match[2]}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 dark:text-blue-400 hover:underline font-medium"
          >
            {match[1]}
          </a>
        );
      }
    }
    return part;
  });
}

export function GeminiAgentDrawer({
  open,
  onOpenChange,
  auditReport,
  messages,
  isLoading,
  onSubmitMessage,
  onClearHistory,
}: GeminiAgentDrawerProps) {
  const [inputValue, setInputValue] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when messages or loading state change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isLoading]);

  const handleSend = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputValue.trim() || isLoading) return;
    const msg = inputValue.trim();
    setInputValue("");
    await onSubmitMessage(msg);
  };

  const handleQuickPrompt = async (prompt: string) => {
    if (isLoading) return;
    await onSubmitMessage(prompt);
  };

  const quickPrompts = [
    {
      label: "Explain BSI Score",
      prompt: "Explain my Bias Severity Index (BSI) score and what risks it indicates.",
    },
    {
      label: "How to fix Demographic Parity?",
      prompt: "What are the recommended steps to fix the Demographic Parity gap in this dataset?",
    },
    {
      label: "Analyze Proxy Risks",
      prompt: "Which proxy columns are driving bias and how do I address them?",
    },
    {
      label: "Action Plan",
      prompt: "Generate a step-by-step mitigation implementation guide for my recommendations.",
    },
  ];

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="flex flex-col h-full w-full max-w-lg p-0 border-l shadow-2xl relative">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100 dark:border-slate-800">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/10 flex items-center justify-center text-indigo-600 dark:text-indigo-400">
              <Sparkles size={18} className="animate-pulse" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
                Gemini Bias Auditor
              </h2>
              <p className="text-xs text-slate-500">Embedded native AI auditor sidekick</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {messages.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onClearHistory}
                className="h-8 w-8 rounded-full p-0"
                title="Clear conversation"
              >
                <RefreshCw size={14} className="text-slate-400 hover:text-slate-600" />
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onOpenChange(false)}
              className="h-8 w-8 rounded-full p-0"
            >
              <X size={16} className="text-slate-400 hover:text-slate-600" />
            </Button>
          </div>
        </div>

        {/* Message Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-slate-50/50 dark:bg-slate-900/20">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center max-w-sm mx-auto space-y-4 animate-fade-rise">
              <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 flex items-center justify-center text-indigo-600 dark:text-indigo-400 mb-2">
                <Sparkles size={24} />
              </div>
              <h3 className="text-sm font-semibold text-slate-950 dark:text-slate-100">
                Meet your AI Auditing Sidekick
              </h3>
              <p className="text-xs leading-5 text-slate-500">
                I have full context of this fairness audit. Ask me to explain the parity metrics,
                analyze proxy attributes, or design mitigation strategies tailored to your dataset.
              </p>

              <div className="w-full pt-4 space-y-2">
                <p className="text-[10px] uppercase font-bold tracking-wider text-slate-400 text-left mb-2">
                  Suggested actions
                </p>
                {quickPrompts.map((item) => (
                  <button
                    key={item.label}
                    onClick={() => handleQuickPrompt(item.prompt)}
                    className="w-full text-left p-3 text-xs bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:border-indigo-500 dark:hover:border-indigo-400 rounded-xl transition shadow-sm hover:shadow text-slate-700 dark:text-slate-200 font-medium"
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((msg, idx) => {
                const isUser = msg.role === "user";
                return (
                  <div
                    key={idx}
                    className={`flex items-start gap-3 max-w-[85%] ${
                      isUser ? "ml-auto flex-row-reverse" : "mr-auto"
                    }`}
                  >
                    <div
                      className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0 ${
                        isUser
                          ? "bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300"
                          : "bg-indigo-100 dark:bg-indigo-900/50 text-indigo-600 dark:text-indigo-400"
                      }`}
                    >
                      {isUser ? <User size={14} /> : <Bot size={14} />}
                    </div>
                    <div
                      className={`p-3.5 rounded-2xl shadow-sm leading-relaxed ${
                        isUser
                          ? "bg-indigo-600 text-white rounded-tr-none"
                          : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-tl-none text-slate-850 dark:text-slate-200"
                      }`}
                    >
                      {isUser ? (
                        <p className="text-sm font-medium whitespace-pre-wrap">{msg.content}</p>
                      ) : (
                        <div className="space-y-1">{renderMarkdown(msg.content)}</div>
                      )}
                    </div>
                  </div>
                );
              })}

              {isLoading && (
                <div className="flex items-start gap-3 max-w-[80%] mr-auto">
                  <div className="w-7 h-7 rounded-full bg-indigo-100 dark:bg-indigo-900/50 text-indigo-600 dark:text-indigo-400 flex items-center justify-center shrink-0">
                    <Bot size={14} />
                  </div>
                  <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 p-3.5 rounded-2xl rounded-tl-none shadow-sm flex items-center gap-2">
                    <div className="w-2.5 h-2.5 bg-indigo-500 rounded-full animate-bounce [animation-delay:-0.3s]" />
                    <div className="w-2.5 h-2.5 bg-indigo-500 rounded-full animate-bounce [animation-delay:-0.15s]" />
                    <div className="w-2.5 h-2.5 bg-indigo-500 rounded-full animate-bounce" />
                  </div>
                </div>
              )}
              <div ref={scrollRef} />
            </div>
          )}
        </div>

        {/* Input Bar */}
        <div className="p-4 border-t border-slate-100 dark:border-slate-800 bg-white dark:bg-slate-900">
          {messages.length > 0 && (
            <div className="flex gap-2 overflow-x-auto pb-3 scrollbar-none max-w-full">
              {quickPrompts.slice(0, 3).map((item) => (
                <button
                  key={item.label}
                  onClick={() => handleQuickPrompt(item.prompt)}
                  disabled={isLoading}
                  className="shrink-0 px-3 py-1.5 text-[10px] bg-slate-50 hover:bg-indigo-50 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-655 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:border-indigo-400 rounded-full transition font-semibold"
                >
                  {item.label}
                </button>
              ))}
            </div>
          )}
          <form onSubmit={handleSend} className="flex gap-2 relative items-center">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask Gemini to explain or suggest mitigations..."
              disabled={isLoading}
              className="flex-1 h-11 pl-4 pr-12 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-850 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:focus:ring-indigo-400 disabled:opacity-50"
            />
            <Button
              type="submit"
              disabled={!inputValue.trim() || isLoading}
              className="absolute right-1.5 top-1.5 h-8 w-8 rounded-lg p-0 flex items-center justify-center bg-indigo-600 hover:bg-indigo-500"
            >
              <Send size={14} className="text-white" />
            </Button>
          </form>
        </div>
      </SheetContent>
    </Sheet>
  );
}

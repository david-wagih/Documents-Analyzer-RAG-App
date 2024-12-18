"use client";

import React, { useState, useRef, useEffect } from "react";
import { DocumentStatus } from "@/types/chat";
import ReactMarkdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import mermaid from "mermaid";
import ZoomableImage from "./ZoomableImage";

// Initialize mermaid
mermaid.initialize({
  startOnLoad: true,
  theme: "default",
  securityLevel: "loose",
});

interface Message {
  role: "user" | "assistant";
  content: string;
  type?: "text" | "diagram" | "clarification";
  mermaid?: string;
  sources?: Array<{
    content: string;
    metadata: any;
  }>;
  has_context?: boolean;
  suggested_types?: string[];
  needs_clarification?: string;
}

interface ChatInterfaceProps {
  documents: DocumentStatus[];
}

const MermaidDiagram = ({ code }: { code: string }) => {
  const [svg, setSvg] = useState<string>("");
  const uniqueId = useRef(`mermaid-${Math.random().toString(36).substr(2, 9)}`);

  useEffect(() => {
    const renderDiagram = async () => {
      try {
        mermaid.initialize({
          startOnLoad: true,
          theme: "default",
          securityLevel: "loose",
          flowchart: {
            htmlLabels: true,
            curve: "basis",
          },
          themeVariables: {
            fontSize: "16px",
            fontFamily: "Inter, system-ui, sans-serif",
          },
        });

        const { svg } = await mermaid.render(uniqueId.current, code);
        setSvg(svg);
      } catch (error) {
        console.error("Mermaid rendering error:", error);
      }
    };

    renderDiagram();
  }, [code]);

  return (
    <div className="my-6 w-full">
      <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-200">
        {svg && <ZoomableImage svg={svg} />}
      </div>
    </div>
  );
};

export default function ChatInterface({ documents }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    console.log("Documents available for chat:", documents);
  }, [documents]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    console.log("Sending message:", userMessage);
    console.log(
      "Active documents:",
      documents.map((d) => ({ id: d.id, name: d.filename }))
    );

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setIsLoading(true);

    try {
      const payload = {
        message: userMessage,
        documentIds: documents.map((doc) => doc.id),
      };
      console.log("Sending request to API:", payload);

      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      console.log("API Response status:", response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error("API Error:", errorText);
        throw new Error(`Failed to get response: ${errorText}`);
      }

      const data = await response.json();
      console.log("API Response data:", data);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.response,
          type: data.type,
          mermaid: data.mermaid,
          sources: data.sources,
          has_context: data.has_context,
          suggested_types: data.suggested_types,
          needs_clarification: data.needs_clarification,
        },
      ]);
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, I encountered an error processing your request.",
          type: "text",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Add the MarkdownComponents definition
  const MarkdownComponents: Components = {
    p: ({ children }) => <p className="mb-2">{children}</p>,
    ul: ({ children }) => <ul className="list-disc ml-4 mb-2">{children}</ul>,
    ol: ({ children }) => (
      <ol className="list-decimal ml-4 mb-2">{children}</ol>
    ),
    li: ({ children }) => <li className="mb-1">{children}</li>,
    code: ({ className, children }) => {
      const match = /language-(\w+)/.exec(className || "");
      return (
        <code
          className={`${
            !className
              ? "bg-gray-200 rounded px-1"
              : "block bg-gray-200 p-2 rounded"
          }`}
        >
          {children}
        </code>
      );
    },
    pre: ({ children }) => (
      <pre className="bg-gray-200 p-2 rounded mb-2">{children}</pre>
    ),
    h1: ({ children }) => (
      <h1 className="text-xl font-bold mb-2">{children}</h1>
    ),
    h2: ({ children }) => (
      <h2 className="text-lg font-bold mb-2">{children}</h2>
    ),
    h3: ({ children }) => (
      <h3 className="text-md font-bold mb-2">{children}</h3>
    ),
    a: ({ href, children }) => (
      <a
        href={href}
        className="text-blue-600 hover:underline"
        target="_blank"
        rel="noopener noreferrer"
      >
        {children}
      </a>
    ),
    blockquote: ({ children }) => (
      <blockquote className="border-l-4 border-gray-300 pl-4 italic">
        {children}
      </blockquote>
    ),
    table: ({ children }) => (
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-300">
          {children}
        </table>
      </div>
    ),
    th: ({ children }) => <th className="px-3 py-2 bg-gray-200">{children}</th>,
    td: ({ children }) => <td className="px-3 py-2 border-t">{children}</td>,
  };

  return (
    <div className="flex flex-col h-[600px] bg-white rounded-lg shadow-lg">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${
              message.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-[80%] rounded-lg p-3 ${
                message.role === "user"
                  ? "bg-blue-500 text-white"
                  : "bg-gray-100 text-gray-900"
              }`}
            >
              {message.role === "assistant" &&
              message.type === "diagram" &&
              message.mermaid ? (
                <div className="w-full max-w-3xl mx-auto">
                  <ReactMarkdown>{message.content}</ReactMarkdown>
                  <MermaidDiagram code={message.mermaid} />
                </div>
              ) : message.role === "assistant" ? (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  className="prose prose-sm max-w-none dark:prose-invert"
                  components={MarkdownComponents}
                >
                  {message.content}
                </ReactMarkdown>
              ) : message.type === "clarification" ? (
                <div className="space-y-2">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    className="prose prose-sm max-w-none dark:prose-invert"
                    components={MarkdownComponents}
                  >
                    {message.content}
                  </ReactMarkdown>
                  {message.suggested_types &&
                    message.suggested_types.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-2">
                        {message.suggested_types.map((type) => (
                          <button
                            key={type}
                            onClick={() => {
                              const newMessage = `Create a ${type} diagram for ${message.needs_clarification}`;
                              setInput(newMessage);
                            }}
                            className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm hover:bg-blue-200 transition-colors"
                          >
                            {type}
                          </button>
                        ))}
                      </div>
                    )}
                </div>
              ) : (
                message.content
              )}
              {message.role === "assistant" &&
                message.has_context === false && (
                  <div className="text-xs text-gray-500 mt-2">
                    No relevant context found in documents
                  </div>
                )}
              {message.role === "assistant" &&
                message.sources &&
                message.sources.length > 0 && (
                  <div className="text-xs text-gray-500 mt-2">
                    Based on content from:{" "}
                    {message.sources
                      .map((source) => source.metadata.filename)
                      .join(", ")}
                  </div>
                )}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-lg p-3 flex items-center space-x-2">
              <div className="animate-bounce">●</div>
              <div className="animate-bounce delay-100">●</div>
              <div className="animate-bounce delay-200">●</div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <form onSubmit={handleSubmit} className="p-4 border-t">
        <div className="flex space-x-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading}
            className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:opacity-50 transition-colors"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}

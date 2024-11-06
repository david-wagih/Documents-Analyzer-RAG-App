import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import mermaid from "mermaid";
import { DocumentStatus } from "../types/chat";
import ZoomableImage from "./ZoomableImage";

interface Message {
  role: "user" | "assistant";
  content: string;
  type?: "text" | "diagram" | "clarification";
  mermaid?: string;
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
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage,
          documentIds: documents.map((doc) => doc.id),
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response");
      }

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.response,
          type: data.type,
          mermaid: data.mermaid,
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
                <>
                  <ReactMarkdown>{message.content}</ReactMarkdown>
                  <MermaidDiagram code={message.mermaid} />
                </>
              ) : message.role === "assistant" &&
                message.type === "clarification" ? (
                <div className="space-y-2">
                  <ReactMarkdown>{message.content}</ReactMarkdown>
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
              ) : message.role === "assistant" ? (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.content}
                </ReactMarkdown>
              ) : (
                message.content
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

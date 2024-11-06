export interface DocumentStatus {
  id: string;
  filename: string;
  status: "processing" | "ready" | "error";
  error?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  type?: "text" | "diagram" | "clarification";
  mermaid?: string;
  sources?: Array<{
    content: string;
    metadata: any;
  }>;
  has_context?: boolean;
}

export interface ChatResponse {
  response: string;
  type: string;
  mermaid?: string;
  sources?: Array<{
    content: string;
    metadata: any;
  }>;
}

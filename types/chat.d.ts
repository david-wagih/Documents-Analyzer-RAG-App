export interface DocumentStatus {
  id: string;
  filename: string;
  status: "processing" | "ready" | "error";
  error?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  response: string;
  error?: string;
}

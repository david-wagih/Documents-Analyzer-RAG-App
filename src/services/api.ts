import axios from "axios";
import { DocumentStatus } from "../types/chat";

const API_URL = "http://localhost:8000";

const api = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

interface DocumentResponse {
  id: string;
  filename: string;
  status: "processing" | "ready" | "error";
  error?: string;
}

export const uploadHRDocument = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);
  const response = await api.post<DocumentResponse>(
    "/api/hr-extract",
    formData
  );
  return response.data;
};

export const uploadDocument = async (file: File): Promise<DocumentResponse> => {
  const formData = new FormData();
  formData.append("file", file);
  const response = await api.post<DocumentResponse>(
    "/api/doc-upload",
    formData
  );
  return response.data;
};

export const sendChatMessage = async (
  message: string,
  documentIds: string[]
) => {
  const response = await api.post("/api/chat", { message, documentIds });
  return response.data;
};

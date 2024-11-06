import React, { useState } from "react";
import FileUpload from "../components/FileUpload";
import ChatInterface from "../components/ChatInterface";
import { DocumentStatus } from "../types/chat";
import { uploadDocument } from "../services/api";

export default function DocChat() {
  const [documents, setDocuments] = useState<DocumentStatus[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = async (file: File) => {
    setIsUploading(true);
    setError(null);

    try {
      const documentData = await uploadDocument(file);
      setDocuments((prev) => [...prev, documentData]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
      console.error("Upload error:", err);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">Documentation Chat</h1>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1">
          <FileUpload
            onFileSelect={handleFileSelect}
            acceptedTypes={["PDF", "MD", "TXT"]}
          />
          {isUploading && (
            <div className="mt-4 p-4 bg-blue-50 rounded-md">
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-500"></div>
                <p className="text-blue-600">Uploading document...</p>
              </div>
            </div>
          )}
          {error && (
            <div className="mt-4 p-4 bg-red-50 text-red-600 rounded-md">
              {error}
            </div>
          )}
          {documents.length > 0 && (
            <div className="mt-4">
              <h3 className="text-lg font-semibold mb-2">Uploaded Documents</h3>
              <ul className="space-y-2">
                {documents.map((doc) => (
                  <li
                    key={doc.id}
                    className="p-2 bg-gray-50 rounded-md text-sm flex justify-between items-center"
                  >
                    <span className="truncate">{doc.filename}</span>
                    <span
                      className={`text-xs px-2 py-1 rounded-full ${
                        doc.status === "ready"
                          ? "bg-green-100 text-green-800"
                          : doc.status === "processing"
                          ? "bg-yellow-100 text-yellow-800"
                          : "bg-red-100 text-red-800"
                      }`}
                    >
                      {doc.status}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
        <div className="lg:col-span-2">
          <ChatInterface documents={documents} />
        </div>
      </div>
    </div>
  );
}

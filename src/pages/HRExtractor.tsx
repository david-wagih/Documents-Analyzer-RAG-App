import React, { useState, useEffect } from "react";
import FileUpload from "../components/FileUpload";
import FilePreview from "../components/FilePreview";
import ProcessingStatus from "../components/ProcessingStatus";
import ExtractedContent from "../components/ExtractedContent";
import { uploadHRDocument } from "../services/api";

export default function HRExtractor() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [extractedData, setExtractedData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileUrl, setFileUrl] = useState<string>("");

  const handleFileSelect = async (file: File) => {
    setSelectedFile(file);
    setFileUrl(URL.createObjectURL(file));
    setIsProcessing(true);
    setError(null);

    try {
      const data = await uploadHRDocument(file);
      setExtractedData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsProcessing(false);
    }
  };

  useEffect(() => {
    return () => {
      if (fileUrl) {
        URL.revokeObjectURL(fileUrl);
      }
    };
  }, [fileUrl]);

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-[1600px] mx-auto px-4">
        <h1 className="text-3xl font-bold mb-8 text-gray-800">
          HR Letter Extractor
        </h1>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - File Upload and Preview */}
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4 text-gray-700">
                Upload Document
              </h2>
              <FileUpload
                onFileSelect={handleFileSelect}
                acceptedTypes={["PDF", "PNG", "JPG", "JPEG"]}
              />
            </div>

            {selectedFile && fileUrl && (
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h2 className="text-xl font-semibold mb-4 text-gray-700">
                  Document Preview
                </h2>
                <FilePreview file={selectedFile} url={fileUrl} />
              </div>
            )}
          </div>

          {/* Right Column - Extracted Information */}
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4 text-gray-700">
                Extracted Information
              </h2>

              {isProcessing && <ProcessingStatus />}

              {error && (
                <div className="mb-6 p-4 bg-red-50 text-red-600 rounded-md">
                  {error}
                </div>
              )}

              {extractedData && (
                <div className="overflow-y-auto max-h-[calc(100vh-300px)]">
                  <ExtractedContent data={extractedData} />
                </div>
              )}

              {!isProcessing && !error && !extractedData && (
                <div className="text-center py-12 text-gray-500">
                  <svg
                    className="mx-auto h-12 w-12 text-gray-400 mb-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  <p>Upload a document to see extracted information</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

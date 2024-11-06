import React, { useState } from "react";
import { FileUploader } from "react-drag-drop-files";

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  acceptedTypes: string[];
}

export default function FileUpload({
  onFileSelect,
  acceptedTypes,
}: FileUploadProps) {
  const [error, setError] = useState<string>("");

  const handleChange = (file: File) => {
    setError("");
    onFileSelect(file);
  };

  const handleError = (err: Error) => {
    setError(err.message);
  };

  return (
    <div className="w-full">
      <FileUploader
        handleChange={handleChange}
        onTypeError={handleError}
        types={acceptedTypes}
        classes="w-full"
      >
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-indigo-500 transition-colors">
          <p className="text-gray-600">
            Drag and drop your file here, or click to select
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Supported formats: {acceptedTypes.join(", ")}
          </p>
        </div>
      </FileUploader>
      {error && <p className="text-red-500 text-sm mt-2">{error}</p>}
    </div>
  );
}

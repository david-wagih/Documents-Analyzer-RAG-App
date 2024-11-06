import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface ExtractedContentProps {
  data: {
    markdown_output?: string;
    raw_analysis?: Record<string, any>;
    confidence?: number;
  };
}

export default function ExtractedContent({ data }: ExtractedContentProps) {
  if (!data) {
    return <div>No data available</div>;
  }

  return (
    <div className="mt-8 space-y-4">
      {data.markdown_output && (
        <div className="prose prose-blue max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {data.markdown_output}
          </ReactMarkdown>
        </div>
      )}

      {data.raw_analysis && (
        <div className="p-4 bg-gray-50 rounded-lg">
          <h3 className="text-lg font-semibold mb-2">Raw Analysis</h3>
          <pre className="whitespace-pre-wrap text-sm bg-white p-4 rounded">
            {JSON.stringify(data.raw_analysis, null, 2)}
          </pre>
        </div>
      )}

      {data.confidence && (
        <div className="p-4 bg-white rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold">Confidence Score</h3>
            <span className="text-sm font-medium text-gray-600">
              {Math.round(data.confidence * 100)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div
              className="bg-blue-600 h-2.5 rounded-full transition-all duration-500"
              style={{ width: `${data.confidence * 100}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

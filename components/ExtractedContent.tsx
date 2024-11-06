import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface ExtractedContentProps {
  data: {
    text?: string;
    metadata?: Record<string, any>;
    markdown_output?: string;
    raw_analysis?: Record<string, any>;
    confidence?: number;
  };
}

export default function ExtractedContent({ data }: ExtractedContentProps) {
  if (!data) {
    return <div>No data available</div>;
  }

  // If we have markdown output, render it with ReactMarkdown
  if (data.markdown_output) {
    return (
      <div className="mt-8 space-y-4">
        <div className="p-6 bg-white rounded-lg shadow-lg">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              h2: ({ node, ...props }) => (
                <h2
                  className="text-2xl font-bold mb-6 text-gray-800 border-b pb-2"
                  {...props}
                />
              ),
              h3: ({ node, ...props }) => (
                <h3
                  className="text-xl font-semibold mt-6 mb-4 text-gray-700"
                  {...props}
                />
              ),
              ul: ({ node, ...props }) => (
                <ul className="space-y-2" {...props} />
              ),
              li: ({ node, ...props }) => (
                <li className="flex items-start space-x-2" {...props} />
              ),
              p: ({ node, ...props }) => (
                <p className="text-gray-600 my-2" {...props} />
              ),
              strong: ({ node, ...props }) => (
                <strong className="text-gray-700 font-semibold" {...props} />
              ),
              em: ({ node, ...props }) => (
                <em className="text-gray-500 italic" {...props} />
              ),
              hr: ({ node, ...props }) => (
                <hr className="my-6 border-gray-200" {...props} />
              ),
              a: ({ node, href, ...props }) => (
                <a
                  href={href}
                  className="text-blue-600 hover:text-blue-800 underline"
                  target="_blank"
                  rel="noopener noreferrer"
                  {...props}
                />
              ),
            }}
            className="prose prose-blue max-w-none"
          >
            {data.markdown_output}
          </ReactMarkdown>
        </div>

        {data.raw_analysis && (
          <div className="mt-6 p-4 bg-gray-50 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-2 text-gray-700">
              Raw Analysis
            </h3>
            <pre className="whitespace-pre-wrap text-sm bg-white p-4 rounded">
              {JSON.stringify(data.raw_analysis, null, 2)}
            </pre>
          </div>
        )}

        {data.confidence && (
          <div className="mt-4 p-4 bg-white rounded-lg shadow">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-lg font-semibold text-gray-700">
                Confidence Score
              </h3>
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

  // Fallback to raw data display
  return (
    <div className="mt-8 space-y-4">
      {data.text && (
        <div className="p-4 bg-white rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">Extracted Text</h3>
          <p className="whitespace-pre-wrap">{data.text}</p>
        </div>
      )}

      {data.metadata && Object.keys(data.metadata).length > 0 && (
        <div className="p-4 bg-white rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">Metadata</h3>
          <dl className="grid grid-cols-2 gap-4">
            {Object.entries(data.metadata).map(([key, value]) => (
              <div key={key}>
                <dt className="text-sm font-medium text-gray-500">
                  {key
                    .replace(/_/g, " ")
                    .replace(/\b\w/g, (l) => l.toUpperCase())}
                </dt>
                <dd className="mt-1 text-sm text-gray-900">
                  {typeof value === "object"
                    ? JSON.stringify(value, null, 2)
                    : String(value)}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      )}

      {data.raw_analysis && (
        <div className="p-4 bg-white rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">Raw Analysis</h3>
          <pre className="whitespace-pre-wrap text-sm">
            {JSON.stringify(data.raw_analysis, null, 2)}
          </pre>
        </div>
      )}

      {data.confidence && (
        <div className="p-4 bg-white rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">Confidence Score</h3>
          <div className="flex items-center">
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div
                className="bg-blue-600 h-2.5 rounded-full"
                style={{ width: `${data.confidence * 100}%` }}
              ></div>
            </div>
            <span className="ml-2 text-sm text-gray-600">
              {Math.round(data.confidence * 100)}%
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

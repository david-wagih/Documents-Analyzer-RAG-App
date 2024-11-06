export default function ProcessingStatus() {
  return (
    <div className="mt-4 p-4 bg-blue-50 rounded-md">
      <div className="flex items-center space-x-3">
        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-500"></div>
        <p className="text-blue-600">Processing document...</p>
      </div>
    </div>
  );
}

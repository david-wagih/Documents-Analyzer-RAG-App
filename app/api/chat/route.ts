import { NextResponse } from "next/server";

export async function POST(request: Request) {
  console.log("Chat API route called");

  try {
    const body = await request.json();
    console.log("Received chat request:", body);

    const response = await fetch("http://localhost:8000/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    console.log("Backend API response status:", response.status);

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Backend API error:", errorText);
      throw new Error(`Failed to get response: ${errorText}`);
    }

    const data = await response.json();
    console.log("Backend API response:", data);

    return NextResponse.json(data);
  } catch (error) {
    console.error("Chat API error:", error);
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Failed to process chat request",
      },
      { status: 500 }
    );
  }
}

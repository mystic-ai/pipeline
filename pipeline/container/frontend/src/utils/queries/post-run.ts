import { GetRunResponse, PostRunPayload } from "../../types";

export async function postRun({
  inputs,
}: PostRunPayload): Promise<GetRunResponse> {
  try {
    const res = await fetch("/v4/runs", {
      method: "POST",
      headers: {
        Accept: "/*",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        inputs,
      }),
    });

    if (!res.ok) {
      const errorResponse = await res.json();
      throw new Error(errorResponse.detail || "Unknown error occurred");
    }

    return await res.json();
  } catch (error) {
    console.error("Error in postRun:", error);

    throw error;
  }
}

interface StreamPostRun extends PostRunPayload {
  onNewChunk: (chunk: string) => void;
}
export async function streamPostRun({
  inputs,
  onNewChunk,
}: StreamPostRun): Promise<void> {
  try {
    const response = await fetch("/v4/runs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ inputs }),
    });

    if (response.body) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          console.log("Streaming finished");
          break;
        }
        // Decode the chunk to a string
        const chunkStr = decoder.decode(value);
        onNewChunk(chunkStr);
        console.log("Received chunk:", chunkStr);
      }
    } else {
      console.log("Response body is null");
    }
  } catch (error) {
    console.error("Error during streaming request", error);
  }
}

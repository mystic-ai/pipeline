import { GetRunResponse, PostRunPayload, RunResult } from "../../types";

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
  onNewChunk: (chunk: RunResult) => void;
}
export async function streamPostRun({
  inputs,
  onNewChunk,
}: StreamPostRun): Promise<void> {
  try {
    const response = await fetch("/v4/runs/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ inputs }),
    });
    if (!response.body) {
      console.log("Response body is null");
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    // Variable to accumulate chunks. Needed when chunk is not valid json
    // and need to accumulate chunks until it is
    let accumulatedData = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        console.log("Streaming finished");
        break;
      }
      accumulatedData += decoder.decode(value);

      try {
        // Attempt to parse the accumulated data as JSON
        const chunkData = JSON.parse(accumulatedData);
        // If parsing is successful, call onNewChunk and reset accumulatedData
        onNewChunk(chunkData);
        console.log("Received chunk:", accumulatedData);
        accumulatedData = "";
      } catch (error) {
        // If parsing fails, continue accumulating data
        console.log("Accumulating data for JSON parsing");
      }
    }
  } catch (error) {
    console.error("Error during streaming request", error);
  }
}

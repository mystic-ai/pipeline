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

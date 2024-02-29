import { PostFileResponse } from "../../types";

export async function postFile({
  formData,
}: {
  formData: FormData;
}): Promise<PostFileResponse> {
  try {
    const res = await fetch("/v4/files", {
      method: "POST",
      headers: {
        Accept: "/*",
      },
      body: formData,
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

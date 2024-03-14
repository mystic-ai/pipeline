import { useMemo } from "react";
import { GetPipelineResponse } from "../types";

const useStreamingIndexes = (pipeline: GetPipelineResponse | undefined) => {
  // Ensure pipeline.output_variables is always an array to prevent "Cannot read properties of undefined" error
  const outputVariables = pipeline?.output_variables || [];

  // The indexes of stream type outputs in the outputs array
  const streamOutputIndexes = useMemo(() => {
    return outputVariables.reduce((acc: number[], output, index) => {
      if (output.run_io_type === "stream") {
        acc.push(index);
      }
      return acc;
    }, []);
  }, [outputVariables]); // Use outputVariables as the dependency

  const isStreaming = useMemo(() => {
    return streamOutputIndexes.length > 0;
  }, [streamOutputIndexes]);

  return { isStreaming, streamOutputIndexes };
};

export default useStreamingIndexes;

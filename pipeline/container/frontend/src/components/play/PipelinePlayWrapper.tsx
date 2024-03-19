import React, { useMemo } from "react";
import { useState } from "react";

import { PipelinePlayColumn } from "./PipelinePlayColumn";
import {
  GetRunResponse,
  RunError,
  RunOutput,
  RunResult,
  StreamingMode,
} from "../../types";
import useGetPipeline from "../../hooks/use-get-pipeline";
import { BlockSkeleton } from "../ui/Skeletons/BlockSkeleton";
import {
  PipelinePlaygroundForm,
  PipelinePlaygroundFormSkeleton,
} from "./PipelinePlaygroundForm";
import { EmptyResourceCard } from "../ui/Cards/EmptyResourceCard";
import { PipelineRunOutput } from "./PipelineRunOutput";
import { Code } from "../ui/Code/Code";
import ChatApp from "./chat-app/ChatApp";
import { ButtonToggle } from "../ui/Buttons/ButtonToggle";
import { Button } from "../ui/Buttons/Button";
import PipelineOutputColumn from "./PipelineOutputColumn";
import useStreamingIndexes from "../../hooks/use-streaming-indexes";

export default function PipelinePlayWrapper(): JSX.Element {
  const [loading, setLoading] = useState<boolean>(false);
  const [runOutputs, setRunOuputs] = useState<RunOutput[]>([]);
  const [runErrors, setRunError] = useState<RunError | null>(null);
  const [activeScreen, setActiveScreen] = useState<"form" | "example">("form");
  const [chatAvailable, setChatAvailable] = useState<boolean>(false);
  const [streamingMode, setStreamingMode] = useState<StreamingMode>("append");

  const { data: pipeline, isLoading: isPipelineLoading } = useGetPipeline();

  const pipelineInputs = pipeline?.input_variables || [];

  const { isStreaming, streamOutputIndexes } = useStreamingIndexes(pipeline);

  // Handlers
  function handleSuccessResult(data: RunOutput[] | null) {
    setRunOuputs(data ? data : []);
  }
  function handleErrorResult(error: RunError | null) {
    setRunError(error);
  }
  function handleIsLoading(isLoading: boolean) {
    setLoading(isLoading);
  }
  function handleNewStreamChunk(chunk: RunResult) {
    try {
      setRunOuputs((oldOutputs) => {
        // Create a copy of the current outputs to modify
        let updatedOutputs = [...oldOutputs];

        // Process each output in the chunk
        chunk.outputs?.forEach((newChunkOutput, chunkIdx) => {
          // Get the existing output for this index if it exists
          const existingOutputIndex = oldOutputs.findIndex(
            (_, idx) => idx === chunkIdx
          );
          const appendToPrevChunk =
            existingOutputIndex !== -1 &&
            streamOutputIndexes.includes(chunkIdx) &&
            streamingMode === "append";

          if (appendToPrevChunk) {
            // Append to stream
            const newValue = newChunkOutput.value || "";
            updatedOutputs[existingOutputIndex] = {
              ...updatedOutputs[existingOutputIndex],
              value: updatedOutputs[existingOutputIndex].value
                ? updatedOutputs[existingOutputIndex].value + newValue
                : newValue,
            };
          } else {
            // Add new output/override if it doesn't exist
            updatedOutputs[chunkIdx] = {
              type: newChunkOutput.type,
              value: newChunkOutput.value,
              file: newChunkOutput.file,
            };
          }
        });

        return updatedOutputs;
      });
    } catch (error) {
      console.error("Error parsing chunk:", error);
    }
  }

  function handleRunFinished(run: GetRunResponse) {
    handleIsLoading(false);
    if (run.outputs) {
      handleErrorResult(null);
      handleSuccessResult(run.outputs);
    }
    if (run.error) {
      handleSuccessResult(null);
      handleErrorResult(run.error);
    }
  }

  function handleRunReset() {
    setRunOuputs([]);
    handleErrorResult(null);
    setLoading(false);
  }
  // Effects
  React.useEffect(() => {
    if (pipeline && pipeline?.extras?.model_type === "chat") {
      setChatAvailable(true);
      setActiveScreen("example");
    }
  }, [isPipelineLoading]);

  // Loading skeleton
  if (isPipelineLoading) {
    return (
      <div className="flex flex-col gap-8">
        <BlockSkeleton height={65} className="block max-w-vcol2 vcol2:hidden" />

        <div className="gap-4 inline-flex flex-col vcol2:flex-row">
          <BlockSkeleton height={600} className="max-w-vcol2" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col gap-6">
        {chatAvailable ? (
          <>
            <ButtonToggle>
              <Button
                colorVariant={
                  activeScreen === "example" ? "secondary" : "muted"
                }
                active={activeScreen === "example"}
                onClick={() => setActiveScreen("example")}
                size="sm"
              >
                Chat mode
              </Button>
              <Button
                colorVariant={activeScreen === "form" ? "secondary" : "muted"}
                active={activeScreen === "form"}
                onClick={() => setActiveScreen("form")}
                size="sm"
              >
                Request builder
              </Button>
            </ButtonToggle>
          </>
        ) : null}
        <div className="flex flex-col gap-8 lg:flex-row max-w-full ">
          {pipeline && activeScreen === "example" ? (
            <div className="w-full vcol2:min-w-vcol2 vcol2:w-vcol2 max-w-full vcol2:max-w-vcol2">
              <ChatApp pipeline={pipeline} />
            </div>
          ) : null}
          {activeScreen === "form" ? (
            <div className="vcol-row vcol-row-2 shadow-sm">
              <PipelinePlayColumn title="Inputs" className="vcol-col-first">
                {isPipelineLoading ? (
                  <PipelinePlaygroundFormSkeleton />
                ) : (
                  <>
                    {pipelineInputs && pipelineInputs.length && pipeline ? (
                      <PipelinePlaygroundForm
                        pipeline={pipeline}
                        handleIsLoading={handleIsLoading}
                        isLoading={loading}
                        handleRunReset={handleRunReset}
                        handleRunComplete={handleRunFinished}
                        handleErrorResult={handleErrorResult}
                        handleNewStreamChunk={handleNewStreamChunk}
                        isStreaming={isStreaming}
                      />
                    ) : (
                      <EmptyResourceCard size="sm">
                        Pipeline inputs definition is empty
                      </EmptyResourceCard>
                    )}
                  </>
                )}
              </PipelinePlayColumn>
              <PipelineOutputColumn
                isStreaming={isStreaming}
                onStreamingModeChange={(newMode) => setStreamingMode(newMode)}
              >
                {loading && !isStreaming ? (
                  <BlockSkeleton height={200} />
                ) : (
                  <>
                    {runOutputs && runOutputs.length ? (
                      <PipelineRunOutput outputs={runOutputs} />
                    ) : null}
                  </>
                )}

                {runErrors ? (
                  <div className="flex flex-col gap-4">
                    {runErrors.traceback ? (
                      <Code
                        hasTabs={false}
                        hasCopyButton
                        tabs={[
                          {
                            code: `Traceback:
${runErrors.traceback}`,
                            title: "shell",
                          },
                        ]}
                      />
                    ) : null}

                    {runErrors.message ? (
                      <Code
                        hasTabs={false}
                        hasCopyButton
                        tabs={[
                          {
                            code: `Error:
${runErrors.message}`,
                            title: "shell",
                          },
                        ]}
                      />
                    ) : null}
                  </div>
                ) : null}
              </PipelineOutputColumn>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

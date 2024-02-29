import React from "react";

import { useState } from "react";
import { GetPipelineResponse, GetRunResponse, RunError } from "../../types";
import { useNotification } from "../ui/Notifications/Notifications";
import { DynamicFieldsForm } from "./DynamicFieldsForm";
import { Card } from "../ui/Cards/Card";
import { DescriptionText } from "../ui/Typography/DescriptionText";
import { Button } from "../ui/Buttons/Button";
import { InputSkeleton } from "../ui/Inputs/Input";
import { postRun } from "../../utils/queries/post-run";

interface FormProps {
  pipeline: GetPipelineResponse;
  handleIsLoading: (isLoading: boolean) => void;
  isLoading?: boolean;
  handleRunComplete: (run: GetRunResponse) => void;
  handleRunReset: () => void;
  handleErrorResult?: (error: RunError | null) => void;
}

export function PipelinePlaygroundForm({
  pipeline,
  handleIsLoading,
  isLoading,
  handleRunReset,
  handleRunComplete,
}: FormProps): JSX.Element {
  // State
  const [localError, setLocalError] = useState<string>();
  const [resetLoading, setResetLoading] = useState<boolean>(false);

  // Hooks
  const notification = useNotification();

  // Handlers
  function handleReset() {
    setResetLoading(true);
    setTimeout(() => {
      setResetLoading(false);
    }, 500);

    setLocalError(undefined);
  }

  async function onSubmit(inputs: Array<any>) {
    handleRunReset && handleRunReset();
    handleIsLoading(true);

    postRun({ inputs })
      .then((run) => handleRunComplete(run))
      .catch((error) => {
        console.log(error);
        notification.error({ title: "Error posting run." });
      })
      .finally(() => {
        handleIsLoading(false);
      });
  }

  if (resetLoading) return <PipelinePlaygroundFormSkeleton />;

  return (
    <DynamicFieldsForm
      pipelineInputIOVariables={pipeline.input_variables}
      onSubmitHandler={onSubmit}
    >
      <div className="flex flex-col gap-4 sticky bottom-8">
        {localError && (
          <Card variant="caution" className="flex gap-4 max-w-96 p-2">
            <DescriptionText>{localError}</DescriptionText>
          </Card>
        )}
        <div className="flex gap-4">
          <Button
            colorVariant="primary"
            size="lg"
            type="submit"
            disabled={isLoading}
            loading={isLoading}
          >
            {isLoading ? "Running" : "Run"}
          </Button>

          <Button
            colorVariant="secondary"
            size="lg"
            type="button"
            disabled={resetLoading}
            onClick={() => handleReset()}
          >
            Reset to default
          </Button>
        </div>
      </div>
    </DynamicFieldsForm>
  );
}

export function PipelinePlaygroundFormSkeleton(): JSX.Element {
  return (
    <div className="flex flex-col gap-8 max-w-142">
      <InputSkeleton hintText label />
      <InputSkeleton hintText label />
      <InputSkeleton hintText label />
      <InputSkeleton hintText label />
      <InputSkeleton hintText label />
      <InputSkeleton hintText label />

      <div className="flex flex-col gap-4 sticky bottom-8">
        <div className="flex gap-4">
          <Button colorVariant="primary" size="lg" disabled={true}>
            Run
          </Button>

          <Button
            colorVariant="secondary"
            size="lg"
            type="button"
            disabled
            loading
          >
            Resetting
          </Button>
        </div>
      </div>
    </div>
  );
}

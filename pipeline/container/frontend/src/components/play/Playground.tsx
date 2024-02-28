import React from "react";
import useGetPipeline from "../../hooks/use-get-pipeline";
import { DynamicFieldsForm } from "./DynamicFieldsForm";

export const Playground = (): JSX.Element => {
  const { data: pipeline } = useGetPipeline();
  return (
    <DynamicFieldsForm
      onSubmitHandler={(inputs) => {
        console.log("submitted");
      }}
      pipelineInputIOVariables={pipeline ? pipeline.input_variables : []}
    />
  );
};

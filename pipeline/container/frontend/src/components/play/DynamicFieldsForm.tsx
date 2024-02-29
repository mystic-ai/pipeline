import React from "react";
import { PropsWithChildren, useMemo } from "react";
import { FormProvider, useForm } from "react-hook-form";
import { useNotification } from "../ui/Notifications/Notifications";
import { IOVariable } from "../../types";
import {
  generateDictFromDynamicFields,
  generateDynamicFieldsFromIOVariables,
  generateFormDefaultValues,
} from "../../utils/ioVariables";
import {
  DynamicRunInputList,
  DynamicRunInputListVariantProps,
} from "./DynamicRunInputList";

interface Props extends PropsWithChildren {
  pipelineInputIOVariables: IOVariable[];
  onSubmitHandler: (inputs: Array<any>) => Promise<void>;
  className?: string;
  variant?: DynamicRunInputListVariantProps;
}

export function DynamicFieldsForm({
  pipelineInputIOVariables,
  onSubmitHandler,
  className = "flex flex-col gap-8 max-w-142",
  children,
  variant,
}: Props): JSX.Element {
  const notification = useNotification();

  const dynamicFields = useMemo(
    () => generateDynamicFieldsFromIOVariables(pipelineInputIOVariables),
    [pipelineInputIOVariables]
  );

  const formMethods = useForm({
    defaultValues: generateFormDefaultValues(dynamicFields),
  });

  const {
    handleSubmit,
    formState: { errors, dirtyFields },
  } = formMethods;

  // Handlers
  async function onSubmit(data: object) {
    try {
      const inputData = await generateDictFromDynamicFields({
        dynamicFields: dynamicFields,
        data,
      });
      return onSubmitHandler(inputData);
    } catch (error) {
      notification.error({ title: "Error while uploading the file." });
      // console.log("error :>> ", error);
    }
  }

  return (
    <form onSubmit={handleSubmit(onSubmit)} className={className}>
      <FormProvider {...formMethods}>
        <DynamicRunInputList
          dynamicFields={dynamicFields}
          inputErrors={errors}
          variant={variant}
        />
        {children}
      </FormProvider>
    </form>
  );
}

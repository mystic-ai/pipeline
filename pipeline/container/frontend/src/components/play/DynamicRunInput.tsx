import React from "react";

import { ReactNode, useEffect, useState } from "react";
import {
  Controller,
  FieldErrors,
  FieldValues,
  useFormContext,
} from "react-hook-form";
import { DynamicFieldData, fileRunIOTypes } from "../../types";
import { InputField } from "../ui/Inputs/InputField";
import { DynamicRunInputLabel } from "./DynamicRunInputLabel";
import { Input } from "../ui/Inputs/Input";
import { Select, SelectItem } from "../ui/Inputs/Select";
import { isNullOrUndefined } from "../../utils/ioVariables";
import { IOInputAndSlider } from "../ui/Inputs/io/IOInputAndSlider";
import { Switch } from "../ui/Inputs/Switch";
import { Textarea } from "../ui/Inputs/Textarea";
import { PipelineFileUpload } from "../ui/Inputs/io/PipelineFileUpload";

export const DynamicRunInput = ({
  inputType,
  fieldName,
  label,
  fieldDescription,
  id,
  subType,
  defaultValue,
  options = [],
  config = {},
  min,
  max,
  step,
  errors,
  isDirty,
  decimalPlaces,
  optional,
}: DynamicFieldData & {
  errors?: FieldErrors<FieldValues>;
  isDirty?: boolean;
}) => {
  const { register, setValue, clearErrors, watch, setError, control } =
    useFormContext();

  let hasError = errors && errors[fieldName] ? true : false;
  let errorMessage = hasError
    ? errors && (errors[fieldName]?.message as string)
    : undefined;

  // Handlers
  // Handle JSON input
  const handleJsonTextAreaChange = (e: any) => {
    // If empty, remove errors
    if (e.target.value === "") {
      setValue(fieldName, e.target.value);
      clearErrors(fieldName);
      return;
    }

    // Otherwise, attempt to parse the input as JSON
    try {
      // Attempt to parse the input as JSON
      const parsedValue = JSON.parse(e.target.value);

      // If successful, set the value
      setValue(fieldName, parsedValue);

      // Clear any previous errors
      clearErrors(fieldName);

      // Not necessary return of parsed value
      return parsedValue;
    } catch (error) {
      // If parsing fails, set an error message
      setError(fieldName, {
        type: "validate",
        message: `This field must be valid JSON: ${String(
          error
        ).toLowerCase()}`,
      });
    }
  };

  const requiredErrorMsg = optional ? false : "This field is required";
  // Input (Text)
  if (subType === "string" && options.length === 0) {
    // We use useEffect to ensure that the form value is updated
    // whenever the selected option changes. Without it, nested values
    // are being lost
    const [selectedValue, setSelectedValue] = useState(defaultValue);

    useEffect(() => {
      setValue(fieldName, selectedValue);
    }, [selectedValue]);
    return (
      <InputField>
        <DynamicRunInputLabel id={id} label={fieldName}>
          string
        </DynamicRunInputLabel>
        <Input
          id={id}
          type="text"
          {...register(fieldName, {
            required: requiredErrorMsg,
          })}
          defaultValue={defaultValue}
          status={hasError ? "invalid" : "clean"}
          invalidText={errorMessage}
          hintText={fieldDescription}
        />
      </InputField>
    );
  }

  // Dropdown (Strings)
  if (subType === "string" && options.length >= 1) {
    // We use useEffect to ensure that the form value is updated
    // whenever the selected option changes. Without it, nested values
    // are being lost
    const [selectedValue, setSelectedValue] = useState(defaultValue);

    useEffect(() => {
      setValue(fieldName, selectedValue);
    }, [selectedValue]);

    return (
      <InputField>
        <DynamicRunInputLabel id={id} label={fieldName}>
          {subType}
        </DynamicRunInputLabel>

        <Select
          {...register(fieldName, config)}
          onValueChange={(value) => setSelectedValue(value)}
          hintText={fieldDescription}
          defaultValue={defaultValue}
        >
          {options.map((o, index) => (
            <SelectItem value={o.value} key={o.label}>
              {o.label}
            </SelectItem>
          ))}
        </Select>
      </InputField>
    );
  }

  // Integer or floating point (Input number + Slider)
  if (subType === "integer" || subType === "fp") {
    let defaultNumber: number = defaultValue ? defaultValue : min ? min : 0;

    const InputNumberLabel = function () {
      return (
        <DynamicRunInputLabel id={id} label={fieldName}>
          {subType}{" "}
          {!isNullOrUndefined(min) && isNullOrUndefined(max) ? (
            <>(minimum {min})</>
          ) : null}
          {isNullOrUndefined(min) && !isNullOrUndefined(max) ? (
            <>(maximum {max})</>
          ) : null}
          {!isNullOrUndefined(min) && !isNullOrUndefined(max) ? (
            <>
              (between {min} and {max})
            </>
          ) : null}
        </DynamicRunInputLabel>
      );
    };

    if (!min && !max) {
      return (
        <InputField>
          <InputNumberLabel />
          <div className="grid grid-cols-[5.9375rem]">
            <Input
              id={id}
              type="number"
              {...register(fieldName, {
                required: requiredErrorMsg,
              })}
              step={!isNullOrUndefined(step) ? step : undefined}
              defaultValue={defaultValue}
              status={hasError ? "invalid" : "clean"}
              invalidText={errorMessage}
              hintText={fieldDescription}
            />
          </div>
        </InputField>
      );
    } else {
      return (
        <InputField>
          <InputNumberLabel />

          <IOInputAndSlider
            name={fieldName}
            id={id}
            fieldName={fieldName}
            inputType={inputType}
            label={fieldName}
            subType={subType}
            onChange={(value) => {
              setValue(fieldName, value);
              clearErrors(fieldName);
            }}
            min={!isNullOrUndefined(min) ? min : undefined}
            max={!isNullOrUndefined(max) ? max : undefined}
            step={!isNullOrUndefined(step) ? step : undefined}
            config={{
              required: requiredErrorMsg,
            }}
            defaultValue={defaultNumber}
            status={isDirty ? (!hasError ? "clean" : "invalid") : "clean"}
            hintText={fieldDescription}
          />
        </InputField>
      );
    }
  }

  // Toggle (Boolean)
  if (subType === "boolean") {
    // We use useEffect to ensure that the form value is updated
    // whenever the selected option changes. Without it, nested values
    // are being lost
    const [selectedValue, setSelectedValue] = useState(defaultValue);

    useEffect(() => {
      setValue(fieldName, selectedValue);
    }, [selectedValue]);

    return (
      <InputField>
        <DynamicRunInputLabel id={id} label={fieldName}>
          bool
        </DynamicRunInputLabel>

        <Switch
          {...register(fieldName, config)}
          onCheckedChange={(value) => {
            setSelectedValue(value);
          }}
          value={!isNullOrUndefined(selectedValue) ? selectedValue : false}
          defaultChecked={defaultValue !== undefined ? defaultValue : false}
          hintText={fieldDescription}
        />
      </InputField>
    );
  }

  // If nested arbitrary dictionary
  if (
    subType === "dictionary" &&
    typeof defaultValue === "object" &&
    !Array.isArray(defaultValue) &&
    defaultValue !== null
  ) {
    const ArbitraryInputs: ReactNode[] = [];
    for (const [key, value] of Object.entries(defaultValue)) {
      ArbitraryInputs.push(
        <InputField key={key}>
          <DynamicRunInputLabel id={id} label={fieldName}>
            Dictionary
          </DynamicRunInputLabel>
          <Controller
            control={control}
            name={fieldName}
            rules={{
              required: "This field is required",
            }}
            render={({ field: { value, ref } }) => (
              <Textarea
                ref={ref}
                onChange={handleJsonTextAreaChange}
                id={id}
                label={fieldName}
                status={hasError ? "invalid" : "clean"}
                invalidText={errorMessage}
                hintText={fieldDescription}
                defaultValue={JSON.stringify(value, null, 4)}
                autoHeight
                isCode
              />
            )}
          />
        </InputField>
      );
    }

    return ArbitraryInputs;
  }

  // File upload for file and pkl types
  if (fileRunIOTypes.includes(subType)) {
    const fileSizeLimit = 250 * 1024 * 1024;

    return (
      <InputField>
        <DynamicRunInputLabel id={id} label={fieldName}>
          {inputType}
        </DynamicRunInputLabel>
        <PipelineFileUpload
          {...register(fieldName, {
            required: requiredErrorMsg,
          })}
          id={id}
          inputType={inputType}
          subType={subType}
          fieldName={fieldName}
          status={hasError ? "invalid" : "clean"}
          invalidText={errorMessage}
          hintText={fieldDescription}
          onChange={(value: File) => {
            setValue(fieldName, value);
            clearErrors(fieldName);
          }}
          fileSizeLimit={fileSizeLimit}
        />
      </InputField>
    );
  }

  if (subType === "array") {
    // Validate that the input valid json
    // we send it along parsed to the backend

    // We use useEffect to ensure that the form value is updated
    // whenever the selected option changes. Without it, nested values
    // are being lost
    const [selectedValue, setSelectedValue] = useState(defaultValue);

    useEffect(() => {
      setValue(fieldName, selectedValue);
    }, [selectedValue]);

    return (
      <InputField>
        <DynamicRunInputLabel id={id} label={fieldName}>
          array
        </DynamicRunInputLabel>

        <Controller
          control={control}
          name={fieldName}
          rules={{
            required: requiredErrorMsg,
          }}
          render={({ field: { value, ref } }) => (
            <Textarea
              ref={ref}
              onChange={handleJsonTextAreaChange}
              id={id}
              status={hasError ? "invalid" : "clean"}
              invalidText={errorMessage}
              hintText={fieldDescription}
              defaultValue={JSON.stringify(defaultValue, null, 4)}
              autoHeight
              isCode
            />
          )}
        />
      </InputField>
    );
  }
};

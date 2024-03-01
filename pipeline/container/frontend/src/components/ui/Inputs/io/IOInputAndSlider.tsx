import React from "react";
import { ChangeEvent, forwardRef, useEffect, useState } from "react";
import { Slider } from "../Slider";
import { useFormContext } from "react-hook-form";
import { HintText } from "../HintText";
import { InputErrorText } from "../InputErrorText";
import { DynamicFieldData } from "../../../../types";
import { InputField } from "../InputField";

interface IOInputAndSliderProps extends DynamicFieldData {
  name: string;
  step?: number;
  onChange: (newValue: number) => void;
  status?: "clean" | "valid" | "invalid";
  hintText?: string;
  invalidText?: string;
}

const inMinRange = (value: number, min?: number) => {
  if (!min) return true;
  return value >= min;
};
const inMaxRange = (value: number, max?: number) => {
  if (!max) return true;
  return value <= max;
};
const isInRange = (value: number, min?: number, max?: number): boolean => {
  if (!inMaxRange(value, max) || !inMinRange(value, min)) return false;
  return true;
};

const isValid = (value: number, min?: number, max?: number, step?: number) => {
  if (!isInRange(value, min, max)) return false;
  if (!step) return true;
  return value % step === 0;
};

export const IOInputAndSlider = forwardRef<
  HTMLInputElement,
  IOInputAndSliderProps
>(
  (
    {
      id,
      name,
      defaultValue,
      fieldName,
      config,
      inputType,
      subType,
      onChange,
      min,
      max,
      step,
      status,
      hintText,
      invalidText,
      ...rest
    }: IOInputAndSliderProps,
    ref
  ) => {
    const { register } = useFormContext();

    // State
    const [inputValue, setInputValue] = useState<number[]>([
      Number(defaultValue || min || 0),
    ]);

    useEffect(() => {
      onChange(inputValue[0]);
    }, [inputValue]);

    const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
      // Maybe display a message if they enter a invalid value?
      const targetValue = Number(e.target.value);

      if (!isValid(targetValue, min, max, step)) return;

      setInputValue([Number(e.target.value)]);
    };

    return (
      <InputField>
        <div className="gap-4 grid grid-cols-[5.9375rem_1fr]">
          <input
            {...rest}
            {...register(fieldName, config)}
            className={`input ${
              status === "invalid" ? "border-error-300" : "border-gray-300"
            }`}
            id={id}
            type="number"
            value={String(inputValue)}
            onChange={handleInputChange}
            step={step}
            min={min}
            max={max}
          />

          <Slider
            defaultValue={inputValue}
            min={min}
            max={max}
            step={step}
            value={inputValue}
            onValueChange={(value: number[]) => {
              setInputValue(value);
            }}
          />
        </div>

        {/* hint message */}
        {status === "clean" || status === "valid" || hintText ? (
          <HintText>{hintText}</HintText>
        ) : null}

        {/* error message */}
        {status === "invalid" ? (
          <InputErrorText>{invalidText}</InputErrorText>
        ) : null}
      </InputField>
    );
  }
);

import React from "react";
import { ReactNode, forwardRef, useState } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { Label } from "./Label";
import { HintText } from "./HintText";
import { InputErrorText } from "./InputErrorText";
import { InputField } from "./InputField";
import { twMerge } from "../../../utils/class-names";
import { IconCheckmark } from "../Icons/IconCheckmark";
import { IconAlertCircle } from "../Icons/IconAlertCircle";
import { IconEyeOff } from "../Icons/IconEyeOff";
import { IconEye } from "../Icons/IconEye";
import { TextSkeleton } from "../Skeletons/TextSkeleton";
import { BlockSkeleton } from "../Skeletons/BlockSkeleton";

const input = cva("input", {
  variants: {
    status: {
      clean: [""],
      valid: ["input-valid"],
      invalid: ["input-invalid"],
    },
  },
  defaultVariants: {
    status: "clean",
  },
});

type InputType = "text" | "password" | "number";

export interface InputTextProps extends VariantProps<typeof input> {
  id: string;
  name: string;
  type: InputType;
  value?: string | number;
  placeholder?: string;
  status?: "clean" | "valid" | "invalid";
  label?: string;
  helpIconText?: string;
  hintText?: ReactNode | string;
  invalidText?: string;
  autoComplete?: string;
  disabled?: boolean;
  defaultValue?: string | number;
  step?: number;
}

export const Input = forwardRef<HTMLInputElement, InputTextProps>(
  (
    {
      id,
      name,
      status,
      label,
      type,
      hintText,
      invalidText,
      disabled,
      step,
      ...rest
    }: InputTextProps,
    ref
  ) => {
    // State
    const [inputType, setInputType] = useState<InputType>(type);

    // Functions
    function TogglePasswordType() {
      if (inputType === "password") setInputType("text");
      if (inputType === "text") setInputType("password");
    }

    return (
      <InputField>
        {/* label */}
        {label ? <Label id={id}>{label}</Label> : null}

        {/* input + icons */}
        <div className="relative">
          {/* input */}
          <input
            type={inputType}
            name={name}
            id={id}
            ref={ref}
            disabled={disabled || false}
            data-disabled={disabled}
            className={twMerge(input({ status }))}
            step={step}
            defaultValue={rest.defaultValue}
            {...rest}
          />

          {/* icons */}
          <div className="absolute flex gap-0 top-0 right-0 pr-2">
            {status === "valid" ? (
              <label htmlFor={id} className="input-floating-icon">
                <IconCheckmark size={16} className="stroke-success-500" />
              </label>
            ) : null}

            {status === "invalid" ? (
              <label htmlFor={id} className="input-floating-icon">
                <IconAlertCircle size={16} className="stroke-error-500" />
              </label>
            ) : null}

            {type === "password" && !disabled ? (
              <button
                onClick={TogglePasswordType}
                type="button"
                className="input-floating-icon group"
                tabIndex={0} // Remove from tab index flow (screen readers won't care for this)
              >
                {inputType === "text" ? (
                  <IconEyeOff
                    size={16}
                    className="stroke-gray-400 group-hover:stroke-gray-500"
                  />
                ) : (
                  <IconEye
                    size={16}
                    className="stroke-gray-400 group-hover:stroke-gray-500"
                  />
                )}
              </button>
            ) : null}
          </div>
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

interface InputSkeletonProps {
  label?: boolean;
  hintText?: boolean;
}

export function InputSkeleton({
  label,
  hintText,
}: InputSkeletonProps): JSX.Element {
  return (
    <InputField>
      {label ? (
        <TextSkeleton>
          <span className="text-sm font-medium">Label text</span>
        </TextSkeleton>
      ) : null}

      <BlockSkeleton height={37.133} />

      {hintText ? (
        <TextSkeleton>
          <span className="text-sm font-normal">Hint text</span>
        </TextSkeleton>
      ) : null}
    </InputField>
  );
}

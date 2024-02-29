import React from "react";

import { cva, type VariantProps } from "class-variance-authority";
import { TextareaHTMLAttributes, forwardRef, useEffect, useRef } from "react";
import { Label } from "./Label";
import { HintText } from "./HintText";
import { TextSkeleton } from "../Skeletons/TextSkeleton";
import { BlockSkeleton } from "../Skeletons/BlockSkeleton";
import { InputErrorText } from "./InputErrorText";
import { InputField } from "./InputField";
import { twMerge } from "../../../utils/class-names";
import { useTextareaAutoHeight } from "../../../hooks/use-textarea-auto-height";

const textarea = cva("textarea", {
  variants: {
    status: {
      clean: [""],
      valid: ["input-valid"],
      invalid: ["input-invalid"],
    },
    isCode: {
      true: ["font-mono", "text-xs"],
    },
  },
  defaultVariants: {
    status: "clean",
  },
});

export interface TextareaProps
  extends VariantProps<typeof textarea>,
    TextareaHTMLAttributes<HTMLTextAreaElement> {
  id: string;
  status?: "clean" | "valid" | "invalid";
  label?: string;
  hintText?: string;
  invalidText?: string;
  value?: string | number;
  autoHeight?: boolean;
  isCode?: boolean;
  readOnly?: boolean;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  (props, forwardedRef) => {
    const {
      id,
      label,
      status,
      hintText,
      invalidText,
      className,
      autoHeight = false,
      isCode,
      ...rest
    } = props;
    const innerRef = useRef<HTMLTextAreaElement>(null);

    useTextareaAutoHeight({
      ref: innerRef,
      autoHeight,
    });

    useEffect(() => {
      // Add keydown handler to prevent tabbing out of textarea
      const handleKeyDown = (e: KeyboardEvent) => {
        if (e.key === "Tab") {
          e.preventDefault();
          const target = e.target as HTMLTextAreaElement;
          const start = target.selectionStart;
          const end = target.selectionEnd;
          target.value =
            target.value.substring(0, start) +
            "\t" +
            target.value.substring(end);
          target.selectionStart = target.selectionEnd = start + 1;
        }
      };
      isCode && innerRef.current?.addEventListener("keydown", handleKeyDown);
      return () => {
        isCode &&
          innerRef.current?.removeEventListener("keydown", handleKeyDown);
      };
    }, [isCode, innerRef]);

    return (
      <InputField>
        {/* label */}
        {label ? <Label id={id}>{label}</Label> : null}

        <textarea
          id={id}
          className={twMerge(textarea({ status, isCode }), className)}
          ref={innerRef}
          disabled={props.disabled || false}
          data-disabled={props.disabled}
          readOnly={props.readOnly || false}
          {...rest}
        />

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

export function TextareaSkeleton({
  label,
  hintText,
}: InputSkeletonProps): JSX.Element {
  return (
    <InputField>
      {/* label + input */}
      {label ? (
        <TextSkeleton>
          <span className="text-sm font-medium">Label text</span>
        </TextSkeleton>
      ) : null}

      <BlockSkeleton height={100} />

      {hintText ? (
        <TextSkeleton>
          <span className="text-sm font-normal">Hint text</span>
        </TextSkeleton>
      ) : null}
    </InputField>
  );
}

import React, { forwardRef, ForwardedRef } from "react";
import * as SelectPrimitive from "@radix-ui/react-select";
import type { SelectProps, SelectItemProps } from "@radix-ui/react-select";
import { IconChevronDown } from "../Icons/IconChevronDown";
import { IconCheckmark } from "../Icons/IconCheckmark";
import { Label } from "./Label";
import { Label as LabelPrimitive } from "@radix-ui/react-label";
import type { PropsWithChildren } from "react";
import { HintText } from "./HintText";
import { cva, type VariantProps } from "class-variance-authority";
import { twMerge } from "../../../utils/class-names";

const select = cva(
  "select-trigger [&>.select-icon]:data-[state=open]:rotate-180",
  {
    variants: {
      status: {
        clean: [""],
        valid: ["input-valid"],
        invalid: ["input-invalid"],
      },
      size: {
        xs: ["!h-[23px]"],
        md: [],
      },
    },
    defaultVariants: {
      status: "clean",
    },
  }
);

interface Props
  extends PropsWithChildren,
    SelectProps,
    VariantProps<typeof select> {
  label?: string;
  hintText?: string;
  placeholder?: string;
  disabled?: boolean;
}

export const Select = forwardRef(
  (props: Props, forwardedRef: ForwardedRef<HTMLButtonElement>) => {
    const {
      label,
      hintText,
      placeholder,
      children,
      status,
      size,
      ...selectProps
    } = props;

    return (
      <LabelPrimitive className="flex flex-col gap-1.5">
        {label ? <Label>{label}</Label> : null}

        <SelectPrimitive.Root {...selectProps}>
          <SelectPrimitive.Trigger
            className={twMerge(select({ status, size }))}
            disabled={props.disabled}
            ref={forwardedRef}
          >
            <SelectPrimitive.Value placeholder={placeholder} />
            <SelectPrimitive.Icon className="select-icon transition">
              <IconChevronDown className="!stroke-gray-500" />
            </SelectPrimitive.Icon>
          </SelectPrimitive.Trigger>

          <SelectPrimitive.Portal>
            <SelectPrimitive.Content
              className="select-content"
              position="popper"
              sideOffset={5}
            >
              <SelectPrimitive.Viewport className="select-viewport">
                {children}
              </SelectPrimitive.Viewport>
            </SelectPrimitive.Content>
          </SelectPrimitive.Portal>

          {/* hint message */}
          {hintText ? <HintText>{hintText}</HintText> : null}
        </SelectPrimitive.Root>
      </LabelPrimitive>
    );
  }
);

export const SelectItem = React.forwardRef(
  ({ children, className, ...props }: SelectItemProps, forwardedRef) => {
    return (
      <SelectPrimitive.Item
        {...props}
        className={twMerge("select-item", className)}
      >
        <SelectPrimitive.ItemText className="select-item-text">
          {children}
        </SelectPrimitive.ItemText>
        <SelectPrimitive.ItemIndicator>
          <IconCheckmark
            className="stroke-primary-600 dark:stroke-primary-300"
            size={20}
          />
        </SelectPrimitive.ItemIndicator>
      </SelectPrimitive.Item>
    );
  }
);

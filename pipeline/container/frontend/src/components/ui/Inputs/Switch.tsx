import React from "react";
import * as SwitchPrimitive from "@radix-ui/react-switch";
import { forwardRef, ElementRef, ComponentPropsWithoutRef } from "react";
import { HintText } from "./HintText";
import { InputField } from "./InputField";
import { twMerge } from "../../../utils/class-names";

interface Props extends ComponentPropsWithoutRef<typeof SwitchPrimitive.Root> {
  hintText?: string;
  invalidText?: string;
  className?: string;
}
const Switch = forwardRef<ElementRef<typeof SwitchPrimitive.Root>, Props>(
  ({ hintText, invalidText, className, ...props }: Props, ref) => (
    <InputField>
      <div className="flex gap-2">
        <SwitchPrimitive.Root
          ref={ref}
          className={twMerge(
            `group w-9 h-5 bg-gray-200 dark:bg-gray-800
            ring-1 ring-gray-300 dark:ring-gray-700
            rounded-full relative cursor-default
            data-[state=checked]:bg-primary-600 dark:data-[state=checked]:bg-primary-500
            outline-none
           hover:bg-gray-300 dark:hover:bg-gray-600 focus-within:ring-2`,
            className
          )}
          {...props}
        >
          <SwitchPrimitive.Thumb
            className="
              block w-4 h-4 bg-white
              rounded-full shadow-sm transition-transform duration-100
              translate-x-0.5 will-change-transform
              group-hover:shadow-md
              data-[state=checked]:translate-x-[1.1875rem]"
          />
        </SwitchPrimitive.Root>
      </div>

      {/* hint message */}
      {hintText ? <HintText>{hintText}</HintText> : null}
    </InputField>
  )
);
Switch.displayName = SwitchPrimitive.Root.displayName;

export { Switch };

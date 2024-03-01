import React from "react";
import { ForwardedRef, forwardRef, PropsWithChildren } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { twMerge } from "tailwind-merge";

const textStyles = cva("text-sm font-normal first-letter:gap-2 text-left", {
  variants: {
    variant: {
      default: ["!text-gray-600", "dark:!text-gray-300"],
      secondary: ["!text-gray-500", "dark:!text-gray-400"],
    },
    defaultVariants: {
      variant: "default",
    },
  },
});

interface Props extends PropsWithChildren, VariantProps<typeof textStyles> {
  className?: string;
  tag?: "p" | "span";
}

export const DescriptionText = forwardRef(
  (
    { children, className, variant = "default", tag }: Props,
    ref: ForwardedRef<HTMLParagraphElement>
  ): JSX.Element => {
    const Component = tag ?? "p";

    return (
      <Component
        ref={ref}
        className={twMerge(textStyles({ variant }), className)}
      >
        {children}
      </Component>
    );
  }
);

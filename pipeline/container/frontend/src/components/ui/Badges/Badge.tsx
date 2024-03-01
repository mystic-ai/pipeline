import { cva, type VariantProps } from "class-variance-authority";
import { PropsWithChildren } from "react";

const badge = cva("border rounded-full text-sm w-fit", {
  variants: {
    variant: {
      default: [""],
      dotted: [
        "border-dotted",
        "border-gray-500",
        "border-gray-500",
        "text-gray-300",
      ],
      primary: ["bg-primary-800", "border-gray-700", "text-primary-200"],
    },
    size: {
      sm: ["px-[5px]", "py-px"],
      base: ["px-[12px]", "py-[6px]"],
      lg: ["px-3", "py-[.4rem]"],
    },
    defaultVariants: {
      variant: "default",
    },
  },
});

interface Props extends VariantProps<typeof badge>, PropsWithChildren {
  tag?: "div" | "span";
}

export function Badge({
  children,
  size = "base",
  variant = "default",
  tag,
}: Props): JSX.Element {
  const Component = tag ?? "div";
  return <Component className={badge({ variant, size })}>{children}</Component>;
}

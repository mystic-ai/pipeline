import React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { PropsWithChildren } from "react";
import { cn } from "../../../utils/class-names";

const card = cva(
  "border rounded gap-2 shadow-sm [&_svg]:shrink-0 [&>hr]:-my-2",
  {
    variants: {
      variant: {
        default: [
          "bg-white",
          "border-gray-200",
          "dark:bg-gray-950",
          "dark:border-gray-700",
          "[&_svg]:stroke-gray-900",
          "dark:[&_svg]:stroke-white",
          "[&>hr]:border-gray-200",
          "dark:[&>hr]:border-gray-700",
        ],

        caution: [
          "bg-[#fffaf4]",
          "border-orange-200",
          "dark:bg-[#682207]",
          "dark:border-[#343024]",
          "text-orange-700",
          "dark:text-orange-100",
          "[&_svg]:stroke-orange-700",
          "dark:[&_svg]:stroke-orange-100",
        ],

        danger: [
          "bg-white",
          "border-error-400",
          "dark:bg-gray-950",
          "dark:border-error-400",
          "[&_svg]:stroke-gray-900",
          "dark:[&_svg]:stroke-white",
          "[&>hr]:border-gray-200",
          "dark:[&>hr]:border-gray-700",
        ],
      },
      size: {
        sm: ["p-2", "text-sm", "[&_svg]:w-4", "[&_svg]:h-4", "[&>hr]:-mx-2"],
        base: ["p-4", "text-base", "[&>hr]:-mx-4"],
      },
      interactive: {
        true: ["transition-colors duration-100 ease-in-out cursor-pointer"],
        false: [],
      },
      defaultVariants: {
        variant: "default",
      },
    },
    compoundVariants: [
      {
        variant: "default",
        interactive: true,
        className: [
          "hover:bg-gray-100",
          "hover:dark:bg-gray-900",
          "hover:border-gray-300",
          "hover:dark:border-gray-500",
          "shadow-sm",
          "group",
        ],
      },
    ],
  }
);

export interface CardProps
  extends VariantProps<typeof card>,
    PropsWithChildren {
  className?: string;
}

export function Card({
  className,
  size = "base",
  variant = "default",
  interactive = false,
  children,
}: CardProps): JSX.Element {
  return (
    <div className={cn(card({ size, variant, interactive }), className)}>
      {children}
    </div>
  );
}

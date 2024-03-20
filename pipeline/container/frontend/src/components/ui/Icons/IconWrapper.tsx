import React from "react";

import { ReactNode } from "react";
import { cva } from "class-variance-authority";
import { twMerge } from "../../../utils/class-names";

const iconWrapperVariants = cva(
  `flex items-center justify-center rounded-full mt-2 [&_svg]:stroke-black dark:[&_svg]:stroke-white`,
  {
    variants: {
      variant: {
        default:
          "bg-primary-100 ring-primary-100/40 dark:bg-gray-700 dark:ring-gray-700/40",
        error:
          "bg-error-100 ring-error-100/40 dark:bg-error-900 dark:ring-error-900/40",
        success:
          "bg-success-100 ring-success-100/40 dark:bg-success-900 dark:ring-success-900/40",
      },
      size: {
        sm: "h-8 w-8 ring-4 [&_svg]:w-4 [&_svg]:h-4",
        base: "h-8 w-8 ring-8 [&_svg]:w-5 [&_svg]:h-5",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "base",
    },
  }
);

export type IconWrapperVariants = "success" | "error" | "default";
interface IconWrapperProps {
  children: ReactNode;
  variant?: IconWrapperVariants;
  className?: string;
  size?: "sm" | "base";
}

export function IconWrapper({
  children,
  variant = "default",
  size,
  className,
}: IconWrapperProps): JSX.Element {
  return (
    <div className={twMerge(iconWrapperVariants({ variant, size }), className)}>
      {children}
    </div>
  );
}

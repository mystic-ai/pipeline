import React from "react";
import { PropsWithChildren } from "react";
import { twMerge } from "../../../utils/class-names";

interface Props extends PropsWithChildren {
  width?: string;
  className?: string;
}

export const TextSkeleton = ({ children, className, width }: Props) => {
  return (
    <span
      style={{ width: width ? `${width}` : "" }}
      className={twMerge("text-skeleton", className)}
    >
      {children}
    </span>
  );
};

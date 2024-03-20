import React from "react";
import { twMerge } from "../../../utils/class-names";

interface BlockSkeletonProps {
  height?: number;
  width?: number;
  className?: string;
}

export const BlockSkeleton = ({
  height,
  width,
  className,
}: BlockSkeletonProps) => {
  return (
    <div
      style={{
        width: width ? `${width}px` : "100%",
        height: height ? `${height}px` : "100%",
      }}
      className={twMerge(
        `
          block-skeleton rounded-lg
        `,
        className
      )}
    ></div>
  );
};

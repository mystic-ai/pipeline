import React from "react";
import { IconTypes } from "./types";

export function IconInfoCircle({
  className,
  size = 24,
}: IconTypes): JSX.Element {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox={`0 0 24 24`}
      role="img"
      fill="none"
      className={className}
    >
      <path
        className={className}
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M12 8.00006V12.0001M12 16.0001H12.01M22 12.0001C22 17.5229 17.5228 22.0001 12 22.0001C6.47715 22.0001 2 17.5229 2 12.0001C2 6.47721 6.47715 2.00006 12 2.00006C17.5228 2.00006 22 6.47721 22 12.0001Z"
      ></path>
    </svg>
  );
}

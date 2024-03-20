import React from "react";
import { IconTypes } from "./types";

export function IconChevronDown({
  className,
  size = 24,
}: IconTypes): JSX.Element {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      fill="none"
      viewBox={`0 0 20 20`}
      className={className}
    >
      <path
        stroke={className}
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.667"
        d="M5 7.5l5 5 5-5"
      ></path>
    </svg>
  );
}

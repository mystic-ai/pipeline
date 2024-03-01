import React from "react";
import { IconTypes } from "./types";

export function IconCheckmarkCircle({ className, size = 24 }: IconTypes) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      fill="none"
      viewBox="0 0 24 24"
      className={className}
    >
      <path
        className={className}
        stroke="inherit"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M22 11.08V12a10 10 0 11-5.93-9.14M22 4L12 14.01l-3-3"
      ></path>
    </svg>
  );
}

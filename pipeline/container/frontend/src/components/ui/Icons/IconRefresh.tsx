import React from "react";
import { IconTypes } from "./types";

export function IconRefresh({ className, size = 24 }: IconTypes): JSX.Element {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox={`0 0 24 24`}
      role="img"
      className={className}
      fill="none"
    >
      <path
        stroke="inherit"
        className={className}
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M1 4v6m0 0h6m-6 0l4.64-4.36A9 9 0 0120.49 9M23 20v-6m0 0h-6m6 0l-4.64 4.36A9.001 9.001 0 013.51 15"
      ></path>
    </svg>
  );
}

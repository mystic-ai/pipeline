import React from "react";
import { IconTypes } from "./types";

export function IconUpload({
  className,
  size = 24,
  ...props
}: IconTypes): JSX.Element {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox={`0 0 24 24`}
      role="img"
      className={className}
      {...props}
    >
      <path
        className={className}
        fill="transparent"
        stroke="inherit"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M21 15v1.2c0 1.68 0 2.52-.327 3.162a3 3 0 01-1.311 1.311C18.72 21 17.88 21 16.2 21H7.8c-1.68 0-2.52 0-3.162-.327a3 3 0 01-1.311-1.311C3 18.72 3 17.88 3 16.2V15m14-7l-5-5m0 0L7 8m5-5v12"
      ></path>
    </svg>
  );
}

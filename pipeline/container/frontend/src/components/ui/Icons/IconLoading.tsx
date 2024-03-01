import React from "react";
import { IconTypes } from "./types";

export function IconLoading({ className, size = 24 }: IconTypes): JSX.Element {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      fill="none"
      viewBox="0 0 24 24"
      className={className}
    >
      <rect
        width="22"
        height="22"
        x="1"
        y="1"
        stroke="url(#paint0_linear_17_2)"
        strokeWidth="2"
        className="fill-transparent"
        rx="11"
      ></rect>
      <defs>
        <linearGradient
          id="paint0_linear_17_2"
          x1="12"
          x2="12"
          y1="0"
          y2="24"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#F0C"></stop>
          <stop offset="1" stopColor="#339"></stop>
        </linearGradient>
      </defs>
    </svg>
  );
}

import React from "react";

import { type IconTypes } from "../Icons/types";

export function PythonLogo({ size = 24 }: IconTypes): JSX.Element {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      fill="none"
      viewBox="0 0 15 16"
    >
      <g clipPath="url(#clip0_482_195299)">
        <path
          fill="url(#paint0_linear_482_195299)"
          className="stroke-0"
          d="M7.554.5c3.808 0 3.57 1.66 3.57 1.66l-.004 1.72H7.486v.516h5.077S15 4.118 15 7.981c0 3.862-2.127 3.725-2.127 3.725h-1.27V9.914s.07-2.138-2.092-2.138H5.907s-2.025.033-2.025-1.967V2.5S3.575.5 7.554.5zm2.004 1.156a.655.655 0 00-.654.658.655.655 0 101.307 0 .655.655 0 00-.653-.658z"
        ></path>
        <path
          className="stroke-0"
          fill="url(#paint1_linear_482_195299)"
          d="M7.447 15.5c-3.807 0-3.57-1.66-3.57-1.66l.005-1.72h3.633v-.516H2.438S.002 11.88.002 8.018s2.126-3.725 2.126-3.725h1.27v1.792S3.329 8.224 5.49 8.224h3.605s2.024-.033 2.024 1.967v3.307s.308 2.002-3.672 2.002zm-2.003-1.157a.655.655 0 00.654-.657.655.655 0 10-1.308 0c0 .363.292.657.654.657z"
        ></path>
      </g>
      <defs>
        <linearGradient
          id="paint0_linear_482_195299"
          x1="13.559"
          x2="6.103"
          y1="1.811"
          y2="9.253"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#387EB8"></stop>
          <stop offset="1" stopColor="#366994"></stop>
        </linearGradient>
        <linearGradient
          id="paint1_linear_482_195299"
          x1="8.997"
          x2="0.992"
          y1="6.57"
          y2="14.2"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#FFE052"></stop>
          <stop offset="1" stopColor="#FFC331"></stop>
        </linearGradient>
        <clipPath id="clip0_482_195299">
          <path
            fill="#fff"
            d="M0 0H15V15H0z"
            transform="matrix(-1 0 0 1 15 .5)"
          ></path>
        </clipPath>
      </defs>
    </svg>
  );
}

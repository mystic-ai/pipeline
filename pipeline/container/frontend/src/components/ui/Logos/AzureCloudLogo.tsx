import React from "react";
import { type IconTypes } from "../Icons/types";

export function AzureCloudLogo({
  width = 33,
  height = 28,
}: IconTypes): JSX.Element {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={width}
      height={height}
      fill="none"
      viewBox="0 0 33 28"
    >
      <g clipPath="url(#clip0_3945_90781)">
        <path
          fill="#0089D6"
          stroke="transparent"
          d="M15.383 26.188a5729.44 5729.44 0 007.585-1.456l.072-.017-3.902-5.033a703.56 703.56 0 01-3.902-5.057c0-.025 4.029-12.057 4.052-12.1.007-.014 2.75 5.12 6.646 12.444l6.684 12.565.051.097-12.4-.002-12.4-.002 7.514-1.44zM.5 24.652c0-.008 1.839-3.469 4.086-7.692l4.085-7.679 4.761-4.333A1844.08 1844.08 0 0118.21.61a.957.957 0 01-.076.21l-5.17 12.026-5.078 11.81-3.693.005c-2.03.003-3.692 0-3.692-.007z"
        ></path>
      </g>
      <defs>
        <clipPath id="clip0_3945_90781">
          <path
            fill="#fff"
            d="M0 0H32V27.2H0z"
            transform="translate(.5 .61)"
          ></path>
        </clipPath>
      </defs>
    </svg>
  );
}

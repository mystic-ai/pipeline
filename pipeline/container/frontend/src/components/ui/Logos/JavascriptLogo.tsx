import React from "react";
import { type IconTypes } from "../Icons/types";

export function JavascriptLogo({ size = 24 }: IconTypes): JSX.Element {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      fill="none"
      viewBox="0 0 15 16"
      className="stroke-0"
    >
      <g clipPath="url(#clip0_482_195308)">
        <path
          fill="#F7DF1E"
          d="M13 .5H2a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-11a2 2 0 00-2-2z"
        ></path>
        <path
          fill="#000"
          d="M10.076 12.217c.302.493.695.855 1.39.855.584 0 .958-.291.958-.695 0-.483-.384-.654-1.027-.935l-.352-.152c-1.017-.433-1.693-.976-1.693-2.123 0-1.058.806-1.862 2.064-1.862.897 0 1.54.312 2.005 1.128l-1.098.705c-.241-.433-.502-.604-.907-.604-.412 0-.674.262-.674.604 0 .423.262.594.866.856l.353.15c1.197.514 1.874 1.038 1.874 2.215 0 1.27-.997 1.964-2.336 1.964-1.31 0-2.155-.623-2.569-1.44l1.146-.666zm-4.98.122c.222.393.423.725.908.725.463 0 .755-.181.755-.886V7.385h1.41v4.812c0 1.46-.856 2.124-2.105 2.124-1.129 0-1.782-.584-2.115-1.288l1.148-.694z"
        ></path>
      </g>
      <defs>
        <clipPath id="clip0_482_195308">
          <path
            fill="#fff"
            d="M0 0H15V15H0z"
            transform="translate(0 .5)"
          ></path>
        </clipPath>
      </defs>
    </svg>
  );
}

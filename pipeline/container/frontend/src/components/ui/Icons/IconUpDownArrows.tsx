import { IconTypes } from "./types";

export const IconUpDownArrows = ({ className, size = 24 }: IconTypes) => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      role="img"
      className={className}
    >
      <path
        className={className}
        stroke="#000"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M8 12.667V3.333m0 0L3.333 8M8 3.333L12.667 8M15.333 11.333v9.334m0 0L20 16m-4.667 4.667L10.667 16"
        fill="transparent"
      ></path>
    </svg>
  );
};

import { IconTypes } from "./types";

export function IconZap({ className, size = 24 }: IconTypes): JSX.Element {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      fill="none"
      viewBox={`0 0 24 24`}
      className={className}
    >
      <path
        stroke={className}
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"
      ></path>
    </svg>
  );
}

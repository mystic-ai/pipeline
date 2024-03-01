import { IconTypes } from "./types";

export function IconCompass({ className, size = 24 }: IconTypes): JSX.Element {
  return (
    <svg
      width={size}
      height={size}
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        className={className}
        stroke="#101828"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"
      ></path>
      <path
        className={className}
        stroke="#101828"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M16.24 7.76l-2.12 6.36-6.36 2.12 2.12-6.36 6.36-2.12z"
      ></path>
    </svg>
  );
}

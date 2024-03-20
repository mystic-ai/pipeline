import { IconTypes } from "./types";

export function IconGlobe({ className, size = 24 }: IconTypes) {
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
        stroke="#101828"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M22 12c0 5.523-4.477 10-10 10m10-10c0-5.523-4.477-10-10-10m10 10H2m10 10C6.477 22 2 17.523 2 12m10 10a15.3 15.3 0 004-10 15.3 15.3 0 00-4-10m0 20a15.3 15.3 0 01-4-10 15.3 15.3 0 014-10M2 12C2 6.477 6.477 2 12 2"
      ></path>
    </svg>
  );
}

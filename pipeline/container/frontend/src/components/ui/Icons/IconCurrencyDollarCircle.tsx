import { IconTypes } from "./types";

export function IconCurrencyDollarCircle({
  className,
  size = 24,
}: IconTypes): JSX.Element {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox={`0 0 24 24`}
      role="img"
      fill="none"
      className={className}
    >
      <path
        stroke="inherit"
        className={className}
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M8.667 14.876A2.333 2.333 0 0011 17.209h2.167a2.5 2.5 0 000-5h-2a2.5 2.5 0 010-5h2.167a2.333 2.333 0 012.333 2.333m-3.5-3.833v1.5m0 10v1.5m10-6.5c0 5.523-4.477 10-10 10s-10-4.477-10-10 4.477-10 10-10 10 4.477 10 10z"
      ></path>
    </svg>
  );
}

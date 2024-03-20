import { IconTypes } from "./types";

export function IconPlus({ className, size = 24 }: IconTypes): JSX.Element {
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
        className={className}
        stroke="inherit"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="3"
        d="M12 5v14m-7-7h14"
      ></path>
    </svg>
  );
}

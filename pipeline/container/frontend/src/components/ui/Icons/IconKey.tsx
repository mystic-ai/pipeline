import { IconTypes } from "./types";

export function IconKey({ className, size = 24 }: IconTypes): JSX.Element {
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
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        fill="transparent"
        d="M15.5 7.5L19 4m2-2l-2 2 2-2zm-9.61 9.61a5.5 5.5 0 11-7.778 7.778 5.5 5.5 0 017.777-7.777l.001-.001zm0 0L15.5 7.5l-4.11 4.11zM15.5 7.5l3 3L22 7l-3-3-3.5 3.5z"
        className={className}
      ></path>
    </svg>
  );
}

import { IconTypes } from "./types";

export function IconRepeat({ className, size = 24 }: IconTypes): JSX.Element {
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
        stroke="#000"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M13 22l-3-3m0 0l3-3m-3 3h5a7 7 0 003-13.326M6 18.326A7 7 0 019 5h5m0 0l-3-3m3 3l-3 3"
      ></path>
    </svg>
  );
}

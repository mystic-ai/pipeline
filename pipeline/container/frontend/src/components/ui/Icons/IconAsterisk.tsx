import { IconTypes } from "./types";

export function IconAsterisk({ className, size = 24 }: IconTypes): JSX.Element {
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
        d="M12 4v16m6-14L6 18m14-6H4m14 6L6 6"
      ></path>
    </svg>
  );
}

import { IconTypes } from "./types";

export function IconGitBranch2({
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
        className={className}
        stroke="#000"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M6 3v12m0 0a3 3 0 103 3m-3-3a3 3 0 013 3m9-9a3 3 0 100-6 3 3 0 000 6zm0 0a9 9 0 01-9 9"
      ></path>
    </svg>
  );
}

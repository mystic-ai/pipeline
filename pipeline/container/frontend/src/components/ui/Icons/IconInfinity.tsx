import { IconTypes } from "./types";

export function IconInfinity({ className, size = 24 }: IconTypes) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      fill="none"
      viewBox="0 0 17 10"
      className={className}
    >
      <path
        className={className}
        stroke="#4E5BA6"
        d="M8.597 5.112C5.862-.46.584-.08.584 4.85c0 4.93 5.389 5.832 8.013.26zm0 0c2.623-5.571 7.82-4.67 7.82 0s-5.086 5.571-7.82 0z"
      ></path>
    </svg>
  );
}

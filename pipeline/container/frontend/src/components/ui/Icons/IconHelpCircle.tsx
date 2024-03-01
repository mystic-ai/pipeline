import { IconTypes } from "./types";

export const IconHelpCircle = ({
  className = "",
  size = 24,
}: IconTypes): JSX.Element => {
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
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3m.08 4h.01M22 12c0 5.523-4.477 10-10 10S2 17.523 2 12 6.477 2 12 2s10 4.477 10 10z"
      ></path>
    </svg>
  );
};

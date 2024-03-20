import { IconTypes } from "./types";

export function IconCoinsStacked({
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
        d="M13.167 5.209c0 1.105-2.462 2-5.5 2s-5.5-.895-5.5-2m11 0c0-1.105-2.462-2-5.5-2s-5.5.895-5.5 2m11 0v1.5m-11-1.5v12c0 1.105 2.462 2 5.5 2m0-8c-.169 0-.335-.003-.5-.008-2.803-.092-5-.949-5-1.992m5.5 6c-3.038 0-5.5-.895-5.5-2m20-1.5c0 1.105-2.462 2-5.5 2s-5.5-.895-5.5-2m11 0c0-1.105-2.462-2-5.5-2s-5.5.895-5.5 2m11 0v7.5c0 1.105-2.462 2-5.5 2s-5.5-.895-5.5-2v-7.5m11 3.75c0 1.105-2.462 2-5.5 2s-5.5-.895-5.5-2"
      ></path>
    </svg>
  );
}

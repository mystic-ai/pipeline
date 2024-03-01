import { IconTypes } from "./types";

export function IconNodePool({
  className,
  size = 24,
  ...props
}: IconTypes): JSX.Element {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox={`0 0 24 24`}
      role="img"
      className={className}
      {...props}
    >
      <path
        className={className}
        fill="transparent"
        stroke="inherit"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M8 3H6.2c-1.12 0-1.68 0-2.108.218a2 2 0 00-.874.874C3 4.52 3 5.08 3 6.2V8m5 13H6.2c-1.12 0-1.68 0-2.108-.218a2 2 0 01-.874-.874C3 19.48 3 18.92 3 17.8V16m18-8V6.2c0-1.12 0-1.68-.218-2.108a2 2 0 00-.874-.874C19.48 3 18.92 3 17.8 3H16m5 13v1.8c0 1.12 0 1.68-.218 2.108a2 2 0 01-.874.874C19.48 21 18.92 21 17.8 21H16m0-9a4 4 0 11-8 0 4 4 0 018 0z"
      ></path>
    </svg>
  );
}

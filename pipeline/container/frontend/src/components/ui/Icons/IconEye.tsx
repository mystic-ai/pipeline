import { IconTypes } from "./types";

export function IconEye({
  className,
  size = 24,
  id = "eye",
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
      <g
        stroke="inherit"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        clipPath={`url(#${id})`}
        className={className}
      >
        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
        <path d="M12 15a3 3 0 100-6 3 3 0 000 6z"></path>
      </g>
      <defs>
        <clipPath id={id}>
          <path fill="#fff" d="M0 0H24V24H0z"></path>
        </clipPath>
      </defs>
    </svg>
  );
}

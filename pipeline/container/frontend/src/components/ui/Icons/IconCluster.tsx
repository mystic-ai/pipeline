import { IconTypes } from "./types";

export function IconCluster({
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
        d="M12 6a1 1 0 100-2 1 1 0 000 2zM12 13a1 1 0 100-2 1 1 0 000 2zM12 20a1 1 0 100-2 1 1 0 000 2zM19 6a1 1 0 100-2 1 1 0 000 2zM19 13a1 1 0 100-2 1 1 0 000 2zM19 20a1 1 0 100-2 1 1 0 000 2zM5 6a1 1 0 100-2 1 1 0 000 2zM5 13a1 1 0 100-2 1 1 0 000 2zM5 20a1 1 0 100-2 1 1 0 000 2z"
      ></path>
    </svg>
  );
}

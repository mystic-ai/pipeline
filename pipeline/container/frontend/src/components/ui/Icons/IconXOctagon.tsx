import { IconTypes } from "./types";

export function IconXOctagon({ className, size = 24 }: IconTypes): JSX.Element {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox={`0 0 24 24`}
      role="img"
      className={className}
    >
      <path
        className={className}
        d="M15.0001 9L9.00006 15M9.00006 9L15.0001 15M7.86006 2H16.1401L22.0001 7.86V16.14L16.1401 22H7.86006L2.00006 16.14V7.86L7.86006 2Z"
        stroke="inherit"
        fill="transparent"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

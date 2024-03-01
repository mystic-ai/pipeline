import { IconTypes } from "./types";

export function IconCoinsHand({
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
        d="M13.53 8.352a3.5 3.5 0 11.941-4.704M6 20.088h2.61c.34 0 .679.04 1.009.12l2.758.67c.598.147 1.222.16 1.826.043l3.05-.593a4.204 4.204 0 002.127-1.107l2.158-2.1a1.503 1.503 0 000-2.168 1.61 1.61 0 00-2.06-.143l-2.515 1.835c-.36.263-.799.405-1.25.405h-2.427 1.545c.871 0 1.577-.687 1.577-1.534v-.307c0-.704-.492-1.317-1.194-1.487l-2.385-.58A5.03 5.03 0 0011.643 13c-.965 0-2.711.799-2.711.799L6 15.025M20 6.5a3.5 3.5 0 11-7 0 3.5 3.5 0 017 0zM2 14.6v5.8c0 .56 0 .84.109 1.054a1 1 0 00.437.437C2.76 22 3.04 22 3.6 22h.8c.56 0 .84 0 1.054-.109a1 1 0 00.437-.437C6 21.24 6 20.96 6 20.4v-5.8c0-.56 0-.84-.109-1.054a1 1 0 00-.437-.437C5.24 13 4.96 13 4.4 13h-.8c-.56 0-.84 0-1.054.109a1 1 0 00-.437.437C2 13.76 2 14.04 2 14.6z"
      ></path>
    </svg>
  );
}

import { IconTypes } from "./types";

export function IconGpu({ className, size = 24 }: IconTypes): JSX.Element {
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
        d="M6 6h.01M6 18h.01m-.81-8h13.6c1.12 0 1.68 0 2.108-.218a2 2 0 00.874-.874C22 8.48 22 7.92 22 6.8V5.2c0-1.12 0-1.68-.218-2.108a2 2 0 00-.874-.874C20.48 2 19.92 2 18.8 2H5.2c-1.12 0-1.68 0-2.108.218a2 2 0 00-.874.874C2 3.52 2 4.08 2 5.2v1.6c0 1.12 0 1.68.218 2.108a2 2 0 00.874.874C3.52 10 4.08 10 5.2 10zm0 12h13.6c1.12 0 1.68 0 2.108-.218a2 2 0 00.874-.874C22 20.48 22 19.92 22 18.8v-1.6c0-1.12 0-1.68-.218-2.108a2 2 0 00-.874-.874C20.48 14 19.92 14 18.8 14H5.2c-1.12 0-1.68 0-2.108.218a2 2 0 00-.874.874C2 15.52 2 16.08 2 17.2v1.6c0 1.12 0 1.68.218 2.108a2 2 0 00.874.874C3.52 22 4.08 22 5.2 22z"
      ></path>
    </svg>
  );
}

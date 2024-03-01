import { IconTypes } from "./types";

export function IconScale({ className, size = 24 }: IconTypes): JSX.Element {
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
        d="M14 22H6.8m0 0c-1.68 0-2.52 0-3.162-.327a3 3 0 01-1.311-1.311C2 19.72 2 18.88 2 17.2M6.8 22h.4c1.68 0 2.52 0 3.162-.327a3 3 0 001.311-1.311C12 19.72 12 18.88 12 17.2v-.4c0-1.68 0-2.52-.327-3.162a3 3 0 00-1.311-1.311C9.72 12 8.88 12 7.2 12h-.4c-1.68 0-2.52 0-3.162.327a3 3 0 00-1.311 1.311C2 14.28 2 15.12 2 16.8v.4m0 0V10m8-8h4m8 8v4m-4 8c.93 0 1.395 0 1.776-.102a3 3 0 002.122-2.122C22 19.395 22 18.93 22 18m0-12c0-.93 0-1.395-.102-1.776a3 3 0 00-2.122-2.122C19.395 2 18.93 2 18 2M6 2c-.93 0-1.395 0-1.776.102a3 3 0 00-2.122 2.122C2 4.605 2 5.07 2 6"
      ></path>
    </svg>
  );
}

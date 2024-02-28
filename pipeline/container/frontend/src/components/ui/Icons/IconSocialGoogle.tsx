import { IconTypes } from "./types";

// Used for Oauth buttons, not meant to customize
export function IconSocialGoogle({
  className,
  size = 24,
}: IconTypes): JSX.Element {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      className={className}
      fill="none"
      viewBox="0 0 32 32"
    >
      <path
        fill="#4285F4"
        className="!stroke-white dark:!stroke-black"
        d="M30.001 16.31c0-1.15-.095-1.99-.301-2.861H16.287v5.195h7.873c-.159 1.291-1.016 3.236-2.92 4.542l-.027.174 4.24 3.22.294.029c2.699-2.443 4.254-6.036 4.254-10.298z"
      ></path>
      <path
        fill="#34A853"
        className="!stroke-white dark:!stroke-black"
        d="M16.285 30c3.857 0 7.095-1.244 9.46-3.391l-4.507-3.423c-1.207.825-2.826 1.4-4.953 1.4a8.584 8.584 0 01-8.127-5.817l-.167.014-4.41 3.344-.058.157C5.873 26.858 10.698 30 16.285 30z"
      ></path>
      <path
        fill="#FBBC05"
        className="!stroke-white dark:!stroke-black"
        d="M8.16 18.769a8.463 8.463 0 01-.476-2.77c0-.964.175-1.897.46-2.768l-.007-.185-4.465-3.399-.146.068A13.785 13.785 0 002.002 16c0 2.256.556 4.387 1.524 6.284L8.16 18.77z"
      ></path>
      <path
        fill="#EB4335"
        className="!stroke-white dark:!stroke-black"
        d="M16.285 7.413c2.683 0 4.492 1.136 5.524 2.085l4.032-3.858C23.365 3.384 20.142 2 16.285 2 10.698 2 5.873 5.142 3.523 9.715l4.62 3.516c1.158-3.375 4.365-5.818 8.142-5.818z"
      ></path>
    </svg>
  );
}

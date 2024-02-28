import { IconTypes } from "./types";

export function IconShieldTick({
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
        className={className}
        stroke="#000"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M9 11.5l2 2L15.5 9m4.5 3c0 4.908-5.354 8.478-7.302 9.615-.221.129-.332.194-.488.227a1.137 1.137 0 01-.42 0c-.156-.033-.267-.098-.488-.227C9.354 20.478 4 16.908 4 12V7.218c0-.8 0-1.2.13-1.543a2 2 0 01.548-.79c.276-.243.65-.383 1.398-.664l5.362-2.01c.208-.078.312-.117.419-.133a1 1 0 01.286 0c.107.016.21.055.419.133l5.362 2.01c.748.281 1.123.421 1.398.665a2 2 0 01.547.789c.131.343.131.743.131 1.543V12z"
      ></path>
    </svg>
  );
}

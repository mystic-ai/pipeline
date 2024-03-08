import { IconTypes } from "./types";

export function IconDiscord({ className, size = 24 }: IconTypes): JSX.Element {
  return (
    <svg
      width={size}
      height={size}
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        className={className}
        stroke="#000"
        d="M7.262 7.406l2.355-1.203a5.087 5.087 0 014.358-.165m-.01.056l.01-.056m0 0l.096-.578c.04-.249.248-.403.45-.337l1.956.644m-2.502.271c.119.05.236.106.353.165l2.355 1.203m-.206-1.639s0 0 0 0zm0 0c.903.298 1.644 1.061 2.032 2.073.055.144.104.294.144.447.498 1.89.878 3.82 1.138 5.778l.19 1.441c.078.582-.093 1.165-.444 1.558-.05.057-.104.11-.161.157l-.186.155a6.78 6.78 0 01-2.302 1.275l-.875.275a.322.322 0 01-.027.007c-.134.03-.272-.035-.354-.168l-.942-1.515M9.977 5.998l-.104-.642c-.04-.251-.246-.408-.446-.34l-1.937.651c-.895.302-1.628 1.074-2.013 2.098a4.438 4.438 0 00-.143.453 48.3 48.3 0 00-1.127 5.848l-.189 1.458c-.087.673.146 1.348.6 1.736l.184.157a6.678 6.678 0 002.28 1.29l.867.279c.141.045.29-.02.378-.163l.932-1.534m7.967-1.57l-.926.67a6.641 6.641 0 01-1.884.96l-.646.205a6.046 6.046 0 01-3.602.027l-.602-.182a6.7 6.7 0 01-2.262-1.209l-.586-.471m7.61-1.094c-1.16 0-1.329-1.313-1.268-1.969 0-.875.724-1.531 1.45-1.531.905 0 1.086 1.094 1.086 1.969s-.543 1.531-1.268 1.531zm-4.71.008c1.149 0 1.316-1.329 1.256-1.993 0-.886-.717-1.55-1.435-1.55-.898 0-1.077 1.107-1.077 1.993 0 .885.538 1.55 1.256 1.55z"
      ></path>
    </svg>
  );
}